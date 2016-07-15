
#define idx(t) \
    ( (int)(t.x + t.y*(tsdf_size + t.z*tsdf_size)) )

#define toVoxel(t) \
    ( (round( ((float)tsdf_size/extent) * t - 0.5f)) )

#define KINECT_MAX_DIST (8.0f)
#define KINECT_MIN_DIST (0.4f)
#define STEP_SIZE (0.002f)

float triLerp (const float3 p, const unsigned int tsdf_size, const float extent, __global const float * tsdf) {

    float3 vox = toVoxel(p);

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

    float3 v000 = (float3)(vox.x + 0, vox.y + 0, vox.z + 0);
    float3 v001 = (float3)(vox.x + 0, vox.y + 0, vox.z + 1);
    float3 v010 = (float3)(vox.x + 0, vox.y + 1, vox.z + 0);
    float3 v011 = (float3)(vox.x + 0, vox.y + 1, vox.z + 1);
    float3 v100 = (float3)(vox.x + 1, vox.y + 0, vox.z + 0);
    float3 v101 = (float3)(vox.x + 1, vox.y + 0, vox.z + 1);
    float3 v110 = (float3)(vox.x + 1, vox.y + 1, vox.z + 0);
    float3 v111 = (float3)(vox.x + 1, vox.y + 1, vox.z + 1);
 
    return \
        tsdf[(int)(idx(v000))] * (1-a) * (1-b) * (1-c) + 
        tsdf[(int)(idx(v001))] * (1-a) * (1-b) * c +
        tsdf[(int)(idx(v010))] * (1-a) * b * (1-c) + 
        tsdf[(int)(idx(v011))] * (1-a) * b * c + 
        tsdf[(int)(idx(v100))] * a * (1-b) * (1-c)+ 
        tsdf[(int)(idx(v101))] * a * (1-b) * c + 
        tsdf[(int)(idx(v110))] * a * b * (1-c) + 
        tsdf[(int)(idx(v111))] * a * b * c;

}


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
    __global const float * T_g_k,
    __global const float * Kinv,
    const float mu,
    const float extent,
    const unsigned int tsdf_size,
    const unsigned int frame_width
 ) {

 	// get the u,v coords of this pixel
	unsigned int u = get_global_id(0);
	unsigned int v = get_global_id(1);

    // now we need to compute the ray from this u,v

    float3 camera_pos = (float3)(T_g_k[3], T_g_k[7], T_g_k[11]);



    float4 uv_sensor;
    uv_sensor.x = Kinv[0]*u + Kinv[1]*v  + Kinv[2];
    uv_sensor.y = Kinv[3]*u + Kinv[4]*v  + Kinv[5];
    uv_sensor.z = Kinv[6]*u + Kinv[7]*v  + Kinv[8];
    uv_sensor.w = 1.0f;
    
    //float4 uv_sensor2 = (float4)((u-640.0f/2.0f)/585.0f, (v-480.0f/2.0f)/585.0f, 1.0f, 1.0f );

    

    float3 uv_world;
    uv_world.x = T_g_k[0]*uv_sensor.x + T_g_k[1]*uv_sensor.y + T_g_k[2]*uv_sensor.z + T_g_k[3]*uv_sensor.w;
    uv_world.y = T_g_k[4]*uv_sensor.x + T_g_k[5]*uv_sensor.y + T_g_k[6]*uv_sensor.z + T_g_k[7]*uv_sensor.w;
    uv_world.z = T_g_k[8]*uv_sensor.x + T_g_k[9]*uv_sensor.y + T_g_k[10]*uv_sensor.z + T_g_k[11]*uv_sensor.w;
    //uv_world.w = T_g_k[12]*uv_sensor.x + T_g_k[13]*uv_sensor.y + T_g_k[14]*uv_sensor.z + T_g_k[15]*uv_sensor.w;

    // compute this ray and normalize it
    float3 ray_dir = normalize(uv_world - camera_pos);

    if (ray_dir.z < 0.0f) {
        vstore3(NAN, v*frame_width + u, vmap);
        return;
    }

    // TODO: Do we need to worry if ray = 0?

    // Sample the TSDF at uv_world position.
    float3 initial_ray = camera_pos + KINECT_MIN_DIST*ray_dir;

    float tsdf_val = tsdf[idx(toVoxel(initial_ray))];
    float tsdf_val_prev;
    float dist;

    for (dist = KINECT_MIN_DIST; dist < KINECT_MAX_DIST; dist += STEP_SIZE) {

        tsdf_val_prev = tsdf_val;

        float3 vox = toVoxel(camera_pos + (dist+STEP_SIZE)*ray_dir);

        // No intersection, outside of the TSDF volume
        if (vox.x < 0 || vox.x > tsdf_size ||
            vox.y < 0 || vox.z > tsdf_size ||
            vox.z < 0 || vox.z > tsdf_size)  {

            break;
        }

        tsdf_val = tsdf[idx(vox)];

        if (tsdf_val > 0.0f && tsdf_val_prev < 0.0f) {
            // Backface
            break;
        }

        if ( tsdf_val < 0.0f && tsdf_val_prev > 0.0f) {
            // Good sign change - we've walked over the level set

            vstore3(camera_pos + (dist+STEP_SIZE) * ray_dir, v*frame_width + u, vmap);
            return;

            // TODO: trilinearly interpolate this!

            float Ftdt = triLerp(camera_pos + (dist+STEP_SIZE) * ray_dir, tsdf_size, extent, tsdf);
            if (isnan(Ftdt)) break;

            float Ft = triLerp(camera_pos + dist*ray_dir, tsdf_size, extent, tsdf);
            if (isnan(Ft)) break;

            float t_star = dist - STEP_SIZE * Ft / (Ftdt - Ft);

            // calculate vertex!
            float3 vertex = camera_pos + ray_dir * t_star;

            vstore3(vertex, v*frame_width + u, vmap);
            return;

            // TODO: get the normal now

        }


        // TODO: Are we going to infinitely loop?
    }

    vstore3(NAN, v*frame_width + u, vmap);
    vstore3(NAN, v*frame_width + u, nmap);

}

