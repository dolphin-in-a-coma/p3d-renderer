#version 150
uniform samplerBuffer matbuf;        // (B*I*4) per-instance model matrix columns // TODO: Check whether it contains the rotation too
uniform samplerBuffer colbuf;        // (B*I)   per-instance color

// Per-view buffers
uniform samplerBuffer viewbuf;       // (K*4) per-view VP matrix
uniform samplerBuffer tilebuf;       // (K)   per-view tile (u0,u1,v0,v1)
uniform int K;                       // number of views

uniform int instancesPerScene;       // I
uniform int shareAcrossScenes;      // 0 = per-scene distinct, 1 = shared across scenes
// TODO: Add lighting?

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

out vec3 v_normal;
out vec4 v_color;
flat out int v_view;
// flat out int v_visible; is not used here
out vec2 v_uv;

void main(){
  int gid = gl_InstanceID;
  int scene = (instancesPerScene>0) ? (gid/instancesPerScene):0;
  int inst = (instancesPerScene>0) ? (gid%instancesPerScene):0;
  int id = (shareAcrossScenes>0) ? inst : gid;
  int mbase = id*4;
  vec4 c0 = texelFetch(matbuf, mbase+0);
  vec4 c1 = texelFetch(matbuf, mbase+1);
  vec4 c2 = texelFetch(matbuf, mbase+2);
  vec4 c3 = texelFetch(matbuf, mbase+3);
  mat4 M = mat4(c0,c1,c2,c3);

  int vbase = scene*4;
  vec4 vc0 = texelFetch(viewbuf, vbase+0);
  vec4 vc1 = texelFetch(viewbuf, vbase+1);
  vec4 vc2 = texelFetch(viewbuf, vbase+2);
  vec4 vc3 = texelFetch(viewbuf, vbase+3);
  mat4 VP = mat4(vc0,vc1,vc2,vc3);

  vec4 clip = VP * (M * p3d_Vertex);

    vec4 tile = texelFetch(tilebuf, scene);
    float x0 = 2.0*tile.x-1.0;
    float x1 = 2.0*tile.y-1.0;
    float y0 = 2.0*tile.z-1.0;
    float y1 = 2.0*tile.w-1.0;
    vec2 s = vec2(0.5*(x1-x0), 0.5*(y1-y0));
    vec2 o = vec2(0.5*(x0+x1), 0.5*(y0+y1));
    gl_Position = vec4(clip.xy*s + o*clip.w, clip.z, clip.w);
  v_normal = normalize(mat3(M)*p3d_Normal);
  v_color = texelFetch(colbuf, id);
  v_view = scene;
  v_uv = p3d_MultiTexCoord0;
}



