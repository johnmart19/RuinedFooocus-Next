// AMD FSR upscaling (FSR 4 when the driver offers it) for one still image, through the AMD FSR API.
// A still has no motion or depth, so the image is presented as a static camera sequence:
// every frame is the source resampled at that frame's sub-pixel jitter, with zero motion
// vectors and a flat depth buffer. Usage:
//   fsr_still in.raw out.raw outW outH [frames] [sign] [versionSubstring] [sharpness]
// Raw files: "RFI1", uint32 width, uint32 height, then RGBA8 rows.
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ffx_api.h"
#include "dx12/ffx_api_dx12.h"
#include "ffx_upscale.h"

using Microsoft::WRL::ComPtr;

static void fail(const char* what, long code = 0)
{
    std::fprintf(stderr, "FAIL %s (0x%lx)\n", what, code);
    std::exit(2);
}
#define CHECK(hr, what)                                                                            \
    do {                                                                                           \
        HRESULT _hr = (hr);                                                                        \
        if (FAILED(_hr)) fail(what, _hr);                                                          \
    } while (0)

static void message(uint32_t type, const wchar_t* text)
{
    std::fwprintf(stderr, L"ffx[%u]: %ls\n", type, text);
}

struct Image
{
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> rgba;
};

static Image readRaw(const char* path)
{
    Image image;
    FILE* f = std::fopen(path, "rb");
    if (!f) fail("open input");
    char magic[4];
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "RFI1", 4)) fail("input magic");
    std::fread(&image.w, 4, 1, f);
    std::fread(&image.h, 4, 1, f);
    image.rgba.resize(size_t(image.w) * image.h * 4);
    if (std::fread(image.rgba.data(), 1, image.rgba.size(), f) != image.rgba.size()) fail("input size");
    std::fclose(f);
    return image;
}

static void writeRaw(const char* path, const Image& image)
{
    FILE* f = std::fopen(path, "wb");
    if (!f) fail("open output");
    std::fwrite("RFI1", 1, 4, f);
    std::fwrite(&image.w, 4, 1, f);
    std::fwrite(&image.h, 4, 1, f);
    std::fwrite(image.rgba.data(), 1, image.rgba.size(), f);
    std::fclose(f);
}

static uint16_t floatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = int32_t((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant = (mant | 0x800000u) >> (1 - exp);
        return uint16_t(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) return uint16_t(sign | 0x7bffu);
    return uint16_t(sign | (uint32_t(exp) << 10) | ((mant + 0x1000u) >> 13));
}

static float lanczos3(float x)
{
    x = std::fabs(x);
    if (x < 1e-6f) return 1.0f;
    if (x >= 3.0f) return 0.0f;
    const float pi = 3.14159265358979323846f;
    const float px = pi * x;
    return (std::sin(px) / px) * (std::sin(px / 3.0f) / (px / 3.0f));
}

// Separable Lanczos-3 resampling in sRGB float, written as RGBA16F (zero 8-bit quantization).
static void jitteredSrgb(
    const std::vector<float>& srgbRgb,
    uint32_t uw,
    uint32_t uh,
    float jx,
    float jy,
    float sign,
    uint8_t* out,
    uint32_t rowPitch,
    std::vector<float>& horiz)
{
    const float sx = -sign * jx, sy = -sign * jy;
    const int bx = int(std::floor(sx)), by = int(std::floor(sy));
    const float fx = sx - float(bx), fy = sy - float(by);
    float wx[6], wy[6], sumX = 0.0f, sumY = 0.0f;
    for (int k = 0; k < 6; ++k) {
        wx[k] = lanczos3(fx - float(k - 2));
        wy[k] = lanczos3(fy - float(k - 2));
        sumX += wx[k];
        sumY += wy[k];
    }
    for (int k = 0; k < 6; ++k) {
        wx[k] /= sumX;
        wy[k] /= sumY;
    }
    const int w = int(uw), h = int(uh);
    horiz.resize(size_t(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        const float* srcRow = srgbRgb.data() + size_t(y) * w * 3;
        float* dstRow = horiz.data() + size_t(y) * w * 3;
        for (int x = 0; x < w; ++x) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (int k = 0; k < 6; ++k) {
                int xx = x + bx + k - 2;
                xx = xx < 0 ? 0 : xx >= w ? w - 1 : xx;
                const float* p = srcRow + size_t(xx) * 3;
                const float wt = wx[k];
                r += wt * p[0];
                g += wt * p[1];
                b += wt * p[2];
            }
            dstRow[size_t(x) * 3 + 0] = r;
            dstRow[size_t(x) * 3 + 1] = g;
            dstRow[size_t(x) * 3 + 2] = b;
        }
    }
    const uint16_t oneHalf = floatToHalf(1.0f);
    for (int y = 0; y < h; ++y) {
        uint16_t* oRow = reinterpret_cast<uint16_t*>(out + size_t(y) * rowPitch);
        int ys[6];
        for (int k = 0; k < 6; ++k) {
            int yy = y + by + k - 2;
            ys[k] = yy < 0 ? 0 : yy >= h ? h - 1 : yy;
        }
        for (int x = 0; x < w; ++x) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (int k = 0; k < 6; ++k) {
                const float* p = horiz.data() + (size_t(ys[k]) * w + x) * 3;
                const float wt = wy[k];
                r += wt * p[0];
                g += wt * p[1];
                b += wt * p[2];
            }
            r = r < 0.0f ? 0.0f : r > 1.0f ? 1.0f : r;
            g = g < 0.0f ? 0.0f : g > 1.0f ? 1.0f : g;
            b = b < 0.0f ? 0.0f : b > 1.0f ? 1.0f : b;
            oRow[size_t(x) * 4 + 0] = floatToHalf(r);
            oRow[size_t(x) * 4 + 1] = floatToHalf(g);
            oRow[size_t(x) * 4 + 2] = floatToHalf(b);
            oRow[size_t(x) * 4 + 3] = oneHalf;
        }
    }
}

static float halfToFloat(uint16_t h)
{
    const uint32_t sign = (h >> 15) & 1, exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
    float value;
    if (exponent == 0) value = std::ldexp(float(mantissa), -24);
    else if (exponent == 31) value = mantissa ? NAN : INFINITY;
    else value = std::ldexp(float(mantissa | 0x400), int(exponent) - 25);
    return sign ? -value : value;
}

struct Gpu
{
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    uint64_t value = 0;

    void run()
    {
        CHECK(list->Close(), "close list");
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        CHECK(queue->Signal(fence.Get(), ++value), "signal");
        CHECK(fence->SetEventOnCompletion(value, event), "fence event");
        if (WaitForSingleObject(event, 60000) != WAIT_OBJECT_0) fail("gpu wait timed out");
        CHECK(device->GetDeviceRemovedReason(), "device removed");
        CHECK(allocator->Reset(), "allocator reset");
        CHECK(list->Reset(allocator.Get(), nullptr), "list reset");
    }
    void transition(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
    {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = r;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = from;
        b.Transition.StateAfter = to;
        list->ResourceBarrier(1, &b);
    }
    ComPtr<ID3D12Resource> texture(uint32_t w, uint32_t h, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = w;
        desc.Height = h;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.Flags = flags;
        ComPtr<ID3D12Resource> r;
        CHECK(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&r)), "create texture");
        return r;
    }
    ComPtr<ID3D12Resource> buffer(uint64_t size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = type;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> r;
        CHECK(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&r)), "create buffer");
        return r;
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint(ID3D12Resource* r, uint64_t* total)
    {
        D3D12_RESOURCE_DESC desc = r->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        device->GetCopyableFootprints(&desc, 0, 1, 0, &fp, nullptr, nullptr, total);
        return fp;
    }
    void copyFromBuffer(ID3D12Resource* texture, ID3D12Resource* upload, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp)
    {
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = texture;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = upload;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = fp;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
};

int main(int argc, char** argv)
{
    if (argc < 5) fail("usage: fsr_still in.raw out.raw outW outH [frames] [sign] [version] [sharpness]");
    const Image source = readRaw(argv[1]);
    const uint32_t outW = uint32_t(std::atoi(argv[3])), outH = uint32_t(std::atoi(argv[4]));
    int frames = argc > 5 ? std::atoi(argv[5]) : 0;
    const float sign = argc > 6 ? float(std::atof(argv[6])) : 1.0f;
    const std::string wanted = argc > 7 ? argv[7] : "";
    const float sharpness = argc > 8 ? float(std::atof(argv[8])) : -1.0f;
    if (outW < source.w || outH < source.h) fail("output must not be smaller than the input");

    std::vector<float> srgbRgb(size_t(source.w) * source.h * 3);
    for (size_t i = 0, n = size_t(source.w) * source.h; i < n; ++i) {
        srgbRgb[i * 3 + 0] = source.rgba[i * 4 + 0] / 255.0f;
        srgbRgb[i * 3 + 1] = source.rgba[i * 4 + 1] / 255.0f;
        srgbRgb[i * 3 + 2] = source.rgba[i * 4 + 2] / 255.0f;
    }
    std::vector<float> horizScratch;

    Gpu gpu;
    ComPtr<IDXGIFactory6> factory;
    CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "dxgi factory");
    ComPtr<IDXGIAdapter1> adapter;
    CHECK(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)), "adapter");
    DXGI_ADAPTER_DESC1 adapterDesc{};
    adapter->GetDesc1(&adapterDesc);
    std::printf("adapter: %ls\n", adapterDesc.Description);
    CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&gpu.device)), "d3d12 device");
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CHECK(gpu.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&gpu.queue)), "queue");
    CHECK(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&gpu.allocator)), "allocator");
    CHECK(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, gpu.allocator.Get(), nullptr, IID_PPV_ARGS(&gpu.list)), "list");
    CHECK(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu.fence)), "fence");
    gpu.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    // Which upscaler versions does this driver and DLL offer?
    uint64_t count = 0;
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = gpu.device.Get();
    versions.outputCount = &count;
    if (ffxQuery(nullptr, &versions.header) != 0) fail("version count");
    std::vector<uint64_t> ids(count);
    std::vector<const char*> names(count);
    versions.versionIds = ids.data();
    versions.versionNames = names.data();
    if (ffxQuery(nullptr, &versions.header) != 0) fail("version list");
    uint64_t chosen = 0;
    for (uint64_t i = 0; i < count; ++i) {
        std::printf("available: %s\n", names[i]);
        if (!chosen && !wanted.empty() && std::string(names[i]).find(wanted) != std::string::npos) chosen = ids[i];
    }
    if (!wanted.empty() && !chosen) fail("requested upscaler version is not available");

    auto color = gpu.texture(source.w, source.h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    auto depth = gpu.texture(source.w, source.h, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    auto motion = gpu.texture(source.w, source.h, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    auto output = gpu.texture(outW, outH, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    uint64_t colorBytes = 0, depthBytes = 0, motionBytes = 0, outputBytes = 0;
    const auto colorFp = gpu.footprint(color.Get(), &colorBytes);
    const auto depthFp = gpu.footprint(depth.Get(), &depthBytes);
    const auto motionFp = gpu.footprint(motion.Get(), &motionBytes);
    const auto outputFp = gpu.footprint(output.Get(), &outputBytes);
    auto colorUpload = gpu.buffer(colorBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto depthUpload = gpu.buffer(depthBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto motionUpload = gpu.buffer(motionBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = gpu.buffer(outputBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    // A flat mid-range depth and zero motion: a static, featureless-depth scene.
    {
        void* data = nullptr;
        CHECK(depthUpload->Map(0, nullptr, &data), "map depth");
        for (uint32_t y = 0; y < source.h; ++y) {
            float* row = reinterpret_cast<float*>(static_cast<uint8_t*>(data) + size_t(y) * depthFp.Footprint.RowPitch);
            for (uint32_t x = 0; x < source.w; ++x) row[x] = 0.5f;
        }
        depthUpload->Unmap(0, nullptr);
        CHECK(motionUpload->Map(0, nullptr, &data), "map motion");
        std::memset(data, 0, size_t(motionBytes));
        motionUpload->Unmap(0, nullptr);
        gpu.copyFromBuffer(depth.Get(), depthUpload.Get(), depthFp);
        gpu.copyFromBuffer(motion.Get(), motionUpload.Get(), motionFp);
        gpu.transition(depth.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.transition(motion.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.run();
    }

    ffxContext context = nullptr;
    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.flags = FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE;
    create.maxRenderSize = {source.w, source.h};
    create.maxUpscaleSize = {outW, outH};
    create.fpMessage = message;
    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = gpu.device.Get();
    ffxCreateContextDescUpscaleVersion apiVersion{};
    apiVersion.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    apiVersion.version = FFX_UPSCALER_VERSION;
    ffxOverrideVersion overrideVersion{};
    overrideVersion.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    overrideVersion.versionId = chosen;
    create.header.pNext = &backend.header;
    backend.header.pNext = &apiVersion.header;
    if (chosen) apiVersion.header.pNext = &overrideVersion.header;
    const ffxReturnCode_t created = ffxCreateContext(&context, &create.header, nullptr);
    if (created != 0) fail("ffxCreateContext", long(created));

    ffxQueryGetProviderVersion provider{};
    provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (ffxQuery(&context, &provider.header) == 0 && provider.versionName) std::printf("provider: %s\n", provider.versionName);

    int32_t phases = 0;
    ffxQueryDescUpscaleGetJitterPhaseCount phaseQuery{};
    phaseQuery.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
    phaseQuery.renderWidth = source.w;
    phaseQuery.displayWidth = outW;
    phaseQuery.pOutPhaseCount = &phases;
    if (ffxQuery(&context, &phaseQuery.header) != 0 || phases <= 0) fail("jitter phase count");
    if (frames <= 0) frames = phases;
    std::printf("jitter phases: %d, frames: %d\n", phases, frames);

    for (int frame = 0; frame < frames; ++frame) {
        float jx = 0, jy = 0;
        ffxQueryDescUpscaleGetJitterOffset jitter{};
        jitter.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET;
        jitter.index = frame;
        jitter.phaseCount = phases;
        jitter.pOutX = &jx;
        jitter.pOutY = &jy;
        if (ffxQuery(&context, &jitter.header) != 0) fail("jitter offset");

        void* data = nullptr;
        CHECK(colorUpload->Map(0, nullptr, &data), "map color");
        jitteredSrgb(srgbRgb, source.w, source.h, jx, jy, sign, static_cast<uint8_t*>(data), colorFp.Footprint.RowPitch, horizScratch);
        colorUpload->Unmap(0, nullptr);
        if (frame) gpu.transition(color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        gpu.copyFromBuffer(color.Get(), colorUpload.Get(), colorFp);
        gpu.transition(color.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        ffxDispatchDescUpscale dispatch{};
        dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dispatch.commandList = gpu.list.Get();
        dispatch.color = ffxApiGetResourceDX12(color.Get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatch.depth = ffxApiGetResourceDX12(depth.Get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatch.motionVectors = ffxApiGetResourceDX12(motion.Get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatch.output = ffxApiGetResourceDX12(output.Get(), FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch.jitterOffset = {jx, jy};
        dispatch.motionVectorScale = {float(source.w), float(source.h)};
        dispatch.renderSize = {source.w, source.h};
        dispatch.upscaleSize = {outW, outH};
        dispatch.enableSharpening = sharpness >= 0;
        dispatch.sharpness = sharpness >= 0 ? sharpness : 0;
        dispatch.frameTimeDelta = 16.6f;
        dispatch.preExposure = 1.0f;
        dispatch.reset = frame == 0;
        dispatch.cameraNear = 0.1f;
        dispatch.cameraFar = 1000.0f;
        dispatch.cameraFovAngleVertical = 1.0f;
        dispatch.viewSpaceToMetersFactor = 1.0f;
        dispatch.flags = FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
        const ffxReturnCode_t dispatched = ffxDispatch(&context, &dispatch.header);
        if (dispatched != 0) fail("ffxDispatch", long(dispatched));
        gpu.run();
    }

    gpu.transition(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    {
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = outputFp;
        src.pResource = output.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    gpu.run();

    Image result;
    result.w = outW;
    result.h = outH;
    result.rgba.resize(size_t(outW) * outH * 4);
    void* data = nullptr;
    CHECK(readback->Map(0, nullptr, &data), "map readback");
    for (uint32_t y = 0; y < outH; ++y) {
        const uint16_t* row = reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(data) + size_t(y) * outputFp.Footprint.RowPitch);
        for (uint32_t x = 0; x < outW; ++x) {
            for (int c = 0; c < 3; ++c) {
                float v = halfToFloat(row[x * 4 + c]) * 255.0f + 0.5f;
                result.rgba[(size_t(y) * outW + x) * 4 + c] = uint8_t(!(v > 0) ? 0 : v > 255 ? 255 : v);
            }
            result.rgba[(size_t(y) * outW + x) * 4 + 3] = 255;
        }
    }
    readback->Unmap(0, nullptr);
    writeRaw(argv[2], result);
    ffxDestroyContext(&context, nullptr);
    std::printf("done %ux%u -> %ux%u\n", source.w, source.h, outW, outH);
    return 0;
}
