#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform float u_float0;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    vec3 color = texture(u_image0, v_texCoord).rgb;
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    fragColor0 = vec4(mix(color, vec3(luminance), u_float0), 1.0);
}
