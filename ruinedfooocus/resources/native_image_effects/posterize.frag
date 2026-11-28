#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform float u_float0;
uniform int u_int0;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    vec3 color = texture(u_image0, v_texCoord).rgb;
    float steps = float(u_int0 - 1);
    vec3 posterized = floor(clamp(color, 0.0, 1.0) * steps + 0.5) / steps;
    fragColor0 = vec4(mix(color, posterized, u_float0), 1.0);
}
