#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform float u_float0;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    vec3 color = texture(u_image0, v_texCoord).rgb;
    vec3 result = pow(clamp(color, 0.0, 1.0), vec3(exp2(-u_float0)));
    fragColor0 = vec4(clamp(result, 0.0, 1.0), 1.0);
}
