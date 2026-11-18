#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform float u_float0;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    vec3 color = texture(u_image0, v_texCoord).rgb;
    float angle = 3.14159265358979 * u_float0;
    vec3 axis = vec3(0.57735026918963);
    vec3 result = color * cos(angle) + cross(axis, color) * sin(angle) + axis * dot(axis, color) * (1.0 - cos(angle));
    fragColor0 = vec4(clamp(result, 0.0, 1.0), 1.0);
}
