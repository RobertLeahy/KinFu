#define KINECT_MAX_DIST (8.0f)
#define KINECT_MIN_DIST (0.4f)
#define STEP_SIZE (0.002f)


float getTsdfValue (const int3 vox, const __global float * tsdf, const size_t size) {

    return tsdf[vox.x + vox.y * size + vox.z * size * size];

}


int3 getVoxel (const float3 pos, const float extent, const size_t size) {

    float flt_size = size;
    float one_over_voxel_size = flt_size / extent;
    return (int3)(
        floor(pos.x * one_over_voxel_size),
        floor(pos.y * one_over_voxel_size),
        floor(pos.z * one_over_voxel_size)
    );

}


int isVoxelValid (const int3 vox, const size_t size) {

    return !(
        (vox.x < 0) || (vox.x >= size) ||
        (vox.y < 0) || (vox.y >= size) ||
        (vox.z < 0) || (vox.z >= size)
    );

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
    const __global float * tsdf,
    __global float * vmap,
    __global float * nmap,
    const __global float * T_g_k,
    const __global float * Kinv,
    const float mu,
    const float extent,
    const unsigned int tsdf_size,
    const unsigned int frame_width
 ) {

	size_t u = get_global_id(0);
	size_t v = get_global_id(1);
    size_t idx = (v * frame_width) + u;

    //  This is where the camera is in world space
    float3 camera_pos = (float3)(T_g_k[3], T_g_k[7], T_g_k[11]);

    //  We now compute where the pixel is in NDC
    float4 uv_sensor;
    uv_sensor.x = Kinv[0]*u + Kinv[1]*v  + Kinv[2];
    uv_sensor.y = Kinv[3]*u + Kinv[4]*v  + Kinv[5];
    uv_sensor.z = Kinv[6]*u + Kinv[7]*v  + Kinv[8];
    uv_sensor.w = 1.0f;

    //  We now compute a point which relative to the camera's
    //  position in world space gives us the direction along
    //  which we must trace to obtain the value for this pixel 
    float3 uv_world;
    uv_world.x = T_g_k[0]*uv_sensor.x + T_g_k[1]*uv_sensor.y + T_g_k[2]*uv_sensor.z + T_g_k[3]*uv_sensor.w;
    uv_world.y = T_g_k[4]*uv_sensor.x + T_g_k[5]*uv_sensor.y + T_g_k[6]*uv_sensor.z + T_g_k[7]*uv_sensor.w;
    uv_world.z = T_g_k[8]*uv_sensor.x + T_g_k[9]*uv_sensor.y + T_g_k[10]*uv_sensor.z + T_g_k[11]*uv_sensor.w;

    //  Obtain the direction and normalize it
    //
    //  TODO: If this is the null vector what happens?  Can
    //  that happen?  Do we need to check?  How would we
    //  handle that?
    float3 ray_dir = normalize(uv_world - camera_pos);
    //  Camera looks in negative Z, this orients it
    //  properly, see: https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
    //  vis-à-vis the update reconstruction pipeline block
    //  implementation
    ray_dir *= (float3)(1, 1, -1);
    //  For some reason without this the image is upside
    //  down
    ray_dir *= (float3)(1, -1, 1);

    //  This gives us the position from which we will begin
    //  our search (i.e. this position is on the near plane)
    float3 initial_ray = camera_pos + KINECT_MIN_DIST * ray_dir;

    //  We trace the ray by starting on the near plane (see
    //  above) and stepping in STEP_SIZE increments until
    //  we find a surface intersection or until we exit the
    //  TSDF volume
    int3 vox = getVoxel(initial_ray, extent, tsdf_size);
    if (!isVoxelValid(vox, tsdf_size)) {

        vstore3(NAN, idx, vmap);
        vstore3(NAN, idx, nmap);

        return;

    }
    float tsdf_val = getTsdfValue(vox, tsdf, tsdf_size);
    //  We have to start STEP_SIZE away because we need
    //  two samples to detect a sign change
    for (float dist = STEP_SIZE; dist < KINECT_MAX_DIST; dist += STEP_SIZE) {

        float3 where = initial_ray + (ray_dir * dist);

        //  Get current TSDF value
        float tsdf_val_prev = tsdf_val;
        vox = getVoxel(where, extent, tsdf_size);
        if (!isVoxelValid(vox, tsdf_size)) break;
        tsdf_val = getTsdfValue(vox, tsdf, tsdf_size);

        int p = signbit(tsdf_val_prev);
        int c = signbit(tsdf_val);
        if (p == c) continue;

        //  Detect backface: From negative to positive
        if (p) break;

        //  WE ARE CURRENTLY APPROXIMATING AND NOT USING
        //  TRILINEAR INTERPOLATION
        //
        //  TODO
        vstore3(where, idx, vmap);
        //  This is bullshit but we need to store something
        //  so we just choose a vector pointing back towards
        //  us
        vstore3(-ray_dir, idx, nmap);

        return;

    }

    vstore3(NAN, idx, vmap);
    vstore3(NAN, idx, nmap);

}

