#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform float u_float0;
uniform float u_float1;
uniform float u_float2;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    vec3 color = texture(u_image0, v_texCoord).rgb;
    vec3 stretched = clamp((color - vec3(u_float1)) / max(u_float2 - u_float1, 0.00001), 0.0, 1.0);
    fragColor0 = vec4(mix(color, stretched, u_float0), 1.0);
}
