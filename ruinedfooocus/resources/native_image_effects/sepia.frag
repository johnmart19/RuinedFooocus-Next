#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform float u_float0;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    vec3 color = texture(u_image0, v_texCoord).rgb;
    vec3 toned = vec3(dot(color, vec3(0.393, 0.769, 0.189)), dot(color, vec3(0.349, 0.686, 0.168)), dot(color, vec3(0.272, 0.534, 0.131)));
    vec3 result = mix(color, clamp(toned, 0.0, 1.0), u_float0);
    fragColor0 = vec4(clamp(result, 0.0, 1.0), 1.0);
}
