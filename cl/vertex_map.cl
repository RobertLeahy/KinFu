#define SMALL_DEPTH (0.1f)

kernel void vertex_map(__global float* src, __global float* dest, __global const float* K_inv, 
const unsigned int width, const unsigned int heights) {

    int x = (int)get_global_id(0); 
    int y = (int)get_global_id(1);

    float depth  = src[y*width + x];

    float3 v;

    if (isnan(depth) || (depth <= SMALL_DEPTH)) {

        v = NAN;

    } else {
    
        // assume z = 1 -> homoegeneous coords
        float v1 = K_inv[0]*x + K_inv[1]*y + K_inv[2];
        float v2 = K_inv[3]*x + K_inv[4]*y + K_inv[5];
        float v3 = K_inv[6]*x + K_inv[7]*y + K_inv[8];
    
        v = depth * (float3)(v1, v2, v3);

    }
    
    vstore3(v, y*width + x, dest);

}