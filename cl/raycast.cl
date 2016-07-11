
#define idx(t) \
    (t.x + t.y*(size + t.z*size)

#define toVoxel(t) \
    (round(((float)tsdf_size/extent) * t - 0.5f))

#define KINECT_MAX_DIST (8.0f)
#define KINECT_MIN_DIST (0.4f)
#define STEP_SIZE (0.8f * mu)

/**
 *	Params:
 *
 *  tsdf - TSDF indexed in the normal way
 *  vmap - Vertex map output
 *  nmap - Normal map output
 *  T_g_k - The T_g_k matrix
 *  Kinv - The inverse K matrix
 *  mu - The TSDF truncation distance
 *  extent - The extent, in meters, of the TSDF (assume square volume here)
 *  tsdf_size - The number of elements in a dimension of the TSDF (assume square volume here)
 *  frame_width - The width of the depth frame
 */
 kernel void raycast(
    __global float * tsdf,
    __global float * vmap,
    __global float * nmap,
    const float * T_g_k,
    const float * Kinv,
    const float mu,
    const float extent,
    const unsigned int size,
    const unsigned int frame_width
 ) {

 	// get the u,v coords of this pixel
	unsigned int u = get_global_id(0);
	unsigned int v = get_global_id(1);

    // now we need to compute the ray from this u,v

    float3 camera_pos = (float3)(T_g_k[3], T_g_k[7], T_g_k[8]);

    float4 uv_sensor;
    uv_sensor.x = K[0]*u + K[1]*v  + K[2];
    uv_sensor.y = K[3]*u + K[4]*v  + K[5];
    uv_sensor.z = K[6]*u + K[7]*v  + K[8];
    uv_sensor.w = 1.0f;

    float3 uv_world;
    uv_world.x = T_g_k[0]*uv_sensor.x + T_g_k[1]*uv_sensor.y + T_g_k[2]*uv_sensor.z + T_g_k[3]*uv_sensor.w;
    uv_world.y = T_g_k[4]*uv_sensor.x + T_g_k[5]*uv_sensor.y + T_g_k[6]*uv_sensor.z + T_g_k[7]*uv_sensor.w;
    uv_world.z = T_g_k[8]*uv_sensor.x + T_g_k[9]*uv_sensor.y + T_g_k[10]*uv_sensor.z + T_g_k[11]*uv_sensor.w;
    //uv_world.w = T_g_k[12]*uv_sensor.x + T_g_k[13]*uv_sensor.y + T_g_k[14]*uv_sensor.z + T_g_k[15]*uv_sensor.w;

    // compute this ray and normalize it
    float ray_dir = normalize(uv_world - camera_pos);


    // TODO: Do we need to worry if ray = 0?


    // Sample the TSDF at uv_world position.
    float3 initial_ray = camera_pos + KINECT_MIN_DIST*ray_dir;

    float tsdf_val = tsdf[idx(toVoxel(initial_ray))];

    for (float dist = KINECT_MIN_DIST; dist < KINECT_MAX_DIST; dist += STEP_SIZE) {

        float tsdf_val_prev = tsdf_val;

        int3 vox = toVoxel(camera_pos + (dist+STEP_SIZE)*ray_dir);

        // No intersection, outside of the TSDF volume
        if (vox.x < 0 || vox.x > tsdf_size
            vox.y < 0 || vox.z > tsdf_size
            vox.z < 0 || vox.z > tsdf_size) break;

        tsdf_val = tsdf[idx(vox)]

        if (tsdf_val > 0.0f && tsdf_val_prev < 0.0f) {
            // Backface
            break;
        }

        if ( tsdf_val < 0.0f && tsdf_val_prev > 0.0f) {
            // Good sign change - we've walked over the level set

            // TODO: trilinearly interpolate this!

            float Ftdt = triLerp(camera_pos + (dist+STEP_SIZE) * ray_dir, tsdf_size, extent, tsdf);
            if (isnan(Ftdt) break;

            float Ft = triLerp(camera_pos + dist*ray_dir, tsdf_size, extent, tsdf);
            if (isnan(Ft)) break;

            float t_star = dist - STEP_SIZE * Ft / (Ftdt - Ft);

            // calculate vertex!
            float3 vertex = camera_pos + ray_dir * t_star;

            // TODO: get the normal now

        }


        // TODO: Are we going to infinitely loop?
    }

    vstore3(vmap, v*frame_width + u, NAN);
    vstore3(nmap, v*frame_width + u, NAN);

}


float triLerp (const float3 p, const unsigned int tsdf_size, const unsigned float extent, const float * tsdf) {

    int3 vox = toVoxel(p);

    float cell_size = extent / tsdf_size;

    if (vox.x <= 0 || vox.x >= tsdf_size - 1) return NAN;
    if (vox.y <= 0 || vox.y >= tsdf_size - 1) return NAN;
    if (vox.z <= 0 || vox.z >= tsdf_size - 1) return NAN;

    float3 vox_world = (vox + 0.5) * cell_size;

    vox.x = (p.x < vox_world.x) ? (vox.x - 1) : vox.x;
    vox.y = (p.y < vox_world.y) ? (vox.y - 1) : vox.y;
    vox.z = (p.z < vox_world.z) ? (vox.z - 1) : vox.z;

    float a = (p.x - (vox.x + 0.5f) * cell_size) / cell_size;
    float b = (p.y - (vox.y + 0.5f) * cell_size) / cell_size;
    float c = (p.z - (vox.z + 0.5f) * cell_size) / cell_size;

    return \
        tsdf[idx(float3)(vox.x + 0, vox.y + 0, vox.z + 0)] * (1-a) * (1-b) * (1-c) +
        tsdf[idx(float3)(vox.x + 0, vox.y + 0, vox.z + 1)] * (1-a) * (1-b) * c +
        tsdf[idx(float3)(vox.x + 0, vox.y + 1, vox.z + 0)] * (1-a) * b * (1-c) +
        tsdf[idx(float3)(vox.x + 0, vox.y + 1, vox.z + 1)] * (1-a) * b * c +
        tsdf[idx(float3)(vox.x + 1, vox.y + 0, vox.z + 0)] * a * (1-b) * (1-c)+
        tsdf[idx(float3)(vox.x + 1, vox.y + 0, vox.z + 1)] * a * (1-b) * c +
        tsdf[idx(float3)(vox.x + 1, vox.y + 1, vox.z + 0)] * a * b * (1-c) +
        tsdf[idx(float3)(vox.x + 1, vox.y + 1, vox.z + 1)] * a *b * c;

}

