#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform float u_float0;
uniform float u_float1;
uniform float u_float2;
uniform float u_float3;
uniform float u_float4;
uniform float u_float5;
uniform float u_float6;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    vec3 color = texture(u_image0, v_texCoord).rgb;
    vec3 low = vec3(u_float1, u_float2, u_float3);
    vec3 high = vec3(u_float4, u_float5, u_float6);
    vec3 balanced = clamp((color - low) / max(high - low, vec3(0.00001)), 0.0, 1.0);
    fragColor0 = vec4(mix(color, balanced, u_float0), 1.0);
}
