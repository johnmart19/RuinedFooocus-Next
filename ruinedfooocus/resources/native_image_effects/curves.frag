#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform sampler2D u_curve0;
uniform sampler2D u_curve1;
uniform sampler2D u_curve2;
uniform sampler2D u_curve3;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
float curveValue(sampler2D curve, float value) {
    return texture(curve, vec2((clamp(value, 0.0, 1.0) * 255.0 + 0.5) / 256.0, 0.5)).r;
}
void main() {
    vec3 color = texture(u_image0, v_texCoord).rgb;
    color = vec3(curveValue(u_curve0, color.r), curveValue(u_curve0, color.g), curveValue(u_curve0, color.b));
    color = vec3(curveValue(u_curve1, color.r), curveValue(u_curve2, color.g), curveValue(u_curve3, color.b));
    fragColor0 = vec4(clamp(color, 0.0, 1.0), 1.0);
}
