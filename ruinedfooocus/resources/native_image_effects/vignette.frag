#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform float u_float0;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    vec3 color = texture(u_image0, v_texCoord).rgb;
    float edge = smoothstep(0.35, 1.0, distance(v_texCoord, vec2(0.5)) * 1.41421356237310);
    vec3 result = color * (1.0 - u_float0 * edge);
    fragColor0 = vec4(clamp(result, 0.0, 1.0), 1.0);
}
