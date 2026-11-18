#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform float u_float0;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
vec3 tap(ivec2 position, ivec2 size, int x, int y) {
    return texelFetch(u_image0, clamp(position + ivec2(x, y), ivec2(0), size - 1), 0).rgb;
}
float luma(vec3 c) {
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}
void main() {
    ivec2 size = textureSize(u_image0, 0);
    ivec2 position = ivec2(v_texCoord * vec2(size));
    vec3 b = tap(position, size, 0, -1), d = tap(position, size, -1, 0), e = tap(position, size, 0, 0);
    vec3 f = tap(position, size, 1, 0), h = tap(position, size, 0, 1);
    vec3 low = min(min(b, d), min(f, h)), high = max(max(b, d), max(f, h));
    float ring = min(min(luma(b), luma(d)), min(luma(f), luma(h)));
    float lower = ring > 0.0 ? clamp(luma(e) / ring, 0.0, 1.0) : 1.0;
    vec3 hitLow = vec3(high.r > 0.0 ? low.r / (4.0 * high.r) : 0.0, high.g > 0.0 ? low.g / (4.0 * high.g) : 0.0, high.b > 0.0 ? low.b / (4.0 * high.b) : 0.0) * lower;
    vec3 bound = 4.0 * low - 4.0;
    vec3 hitHigh = vec3(bound.r < 0.0 ? (1.0 - high.r) / bound.r : 0.0, bound.g < 0.0 ? (1.0 - high.g) / bound.g : 0.0, bound.b < 0.0 ? (1.0 - high.b) / bound.b : 0.0);
    vec3 lobes = max(-hitLow, hitHigh);
    float lobe = max(-0.1875, min(max(lobes.r, max(lobes.g, lobes.b)), 0.0));
    vec3 sharp = clamp((lobe * (b + d + h + f) + e) / (4.0 * lobe + 1.0), 0.0, 1.0);
    fragColor0 = vec4(mix(e, sharp, u_float0), 1.0);
}
