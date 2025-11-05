#version 150
in vec3 v_normal;
in vec4 v_color;
flat in int v_view;
in vec2 v_uv;

out vec4 pbr_FragColor;

uniform sampler2D pbr_Texture0;
uniform float useTexture;        // 0 = no texture, 1 = sample

uniform samplerBuffer tilebuf;   // (K)   per-view tile (u0,u1,v0,v1)
uniform vec2 screenSize;         // (width, height) in pixels

uniform float lightingStrength;  // 0..1
uniform vec3 dirLightDir;
uniform vec3 dirLightCol;
uniform vec3 ambientCol;

void main(){
    // Convert normalized tile rect to pixel bounds
    vec4 tile = texelFetch(tilebuf, v_view);
    float x0 = tile.x * screenSize.x;
    float x1 = tile.y * screenSize.x;
    float y0 = tile.z * screenSize.y;
    float y1 = tile.w * screenSize.y;
    if (gl_FragCoord.x < x0 || gl_FragCoord.x >= x1 || gl_FragCoord.y < y0 || gl_FragCoord.y >= y1) {
        discard;
    }

    vec3 tex = texture(pbr_Texture0, v_uv).rgb;
    vec3 base = mix(vec3(1.0), tex, clamp(useTexture,0.0,1.0));
    vec3 n = normalize(v_normal);
    float ndl = max(dot(n, normalize(dirLightDir)), 0.0);
    vec3 light = ambientCol + ndl*dirLightCol;
    vec3 l = mix(vec3(1.0), light, clamp(lightingStrength,0.0,1.0));
    vec3 col = base * v_color.rgb * l;
    pbr_FragColor = vec4(col, v_color.a);
}



