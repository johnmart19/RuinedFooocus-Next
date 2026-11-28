#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform float u_float0;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
vec3 tap(ivec2 position, ivec2 size, int x, int y) {
    return texelFetch(u_image0, clamp(position + ivec2(x, y), ivec2(0), size - 1), 0).rgb;
}
void order(inout vec3 a, inout vec3 b) {
    vec3 low = min(a, b);
    b = max(a, b);
    a = low;
}
void main() {
    ivec2 size = textureSize(u_image0, 0);
    ivec2 position = ivec2(v_texCoord * vec2(size));
    vec3 v0 = tap(position, size, -1, -1), v1 = tap(position, size, 0, -1), v2 = tap(position, size, 1, -1);
    vec3 v3 = tap(position, size, -1, 0), v4 = tap(position, size, 0, 0), v5 = tap(position, size, 1, 0);
    vec3 v6 = tap(position, size, -1, 1), v7 = tap(position, size, 0, 1), v8 = tap(position, size, 1, 1);
    order(v1, v2); order(v4, v5); order(v7, v8); order(v0, v1); order(v3, v4); order(v6, v7);
    order(v1, v2); order(v4, v5); order(v7, v8); order(v0, v3); order(v5, v8); order(v4, v7);
    order(v3, v6); order(v1, v4); order(v2, v5); order(v4, v7); order(v4, v2); order(v6, v4);
    order(v4, v2);
    fragColor0 = vec4(mix(tap(position, size, 0, 0), v4, u_float0), 1.0);
}
