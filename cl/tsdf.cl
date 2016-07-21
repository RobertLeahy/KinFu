
#define l2_norm(t) \
	( sqrt(t.x*t.x + t.y*t.y + t.z*t.z) )


#define MAX_WEIGHT (256U)
/**
 *	Params:
 *
 *	src - raw (unfiltered) depth image)
 *	dest - allocated TSDF volume
 *	tsdf_width, tsdf_height, tsdf_depth - size params of TSDF
 *	proj_view - proj_view matrix (4x4)
 *	K - camera matrix (3x3)
 *	K_inv - inverse camera matrix (3x3)
 *	t_gk - (vec3) translation component of of T_gk (sesnor pose estimation)
 *	mu - Truncation distance (tunable)
 *	frame_width - width of depth image (for indexing)
 *	frame_height - height of depth image (for indexing)
 *	tsdf_extent_w,h,d - tsdf_extent in m in w,h,d directions
 *
 */
kernel void tsdf_kernel(__global float * src, __global float* dest, 
 const unsigned int tsdf_width, const unsigned int tsdf_height, const unsigned int tsdf_depth,
 __global const float* proj_view, __global const float* K, __global const float* K_inv, 
 __global const float* t_gk, const float mu, const unsigned int frame_width, const unsigned int frame_height, 
 const float tsdf_extent_w, const float tsdf_extent_h, const float tsdf_extent_d, const unsigned int n,
 __global unsigned int * weight) {


 	// get the x, y, z of the current voxel from memory
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	unsigned int z = get_global_id(2);

	// Determine the linear index using the x,y,z components
	// From http://stackoverflow.com/questions/10903149/how-do-i-compute-the-linear-index-of-a-3d-coordinate-and-vice-versa
	unsigned int idx = x + (tsdf_width)*y + ((tsdf_width) * (tsdf_height))*z;


	if (n==0) {
		dest[idx] = 1.0f;
		weight[idx] = 0;
	}

	// OOB
	if (x > tsdf_width || y > tsdf_height || z > tsdf_depth ) {

		 return;
	} 	

	// x,y,z give the voxel id, but we need to compute the position in 3D space.

	// Compute the world coordinates of the center of this voxel
	float p_x = ((float)x + 0.5) * tsdf_extent_w/tsdf_width;
	float p_y = ((float)y + 0.5) * tsdf_extent_h/tsdf_height;
	float p_z = ((float)z + 0.5) * tsdf_extent_d/tsdf_depth;
	float4 p = (float4)(p_x, p_y, p_z, 1);

	// Multiplying by T_g_k.inverse() here. This takes us to camera space.
	float3 cam;
	cam.x = proj_view[0]*p.x + proj_view[1]*p.y + proj_view[2]*p.z + proj_view[3]*p.w;
	cam.y = proj_view[4]*p.x + proj_view[5]*p.y + proj_view[6]*p.z + proj_view[7]*p.w;
	cam.z = proj_view[8]*p.x + proj_view[9]*p.y + proj_view[10]*p.z + proj_view[11]*p.w;

	float3 plane;
	plane.x = cam.x * K[0] + cam.y * K[1] + cam.z * K[2];
	plane.y = cam.x * K[3] + cam.y * K[4] + cam.z * K[5];
	plane.z = cam.x * K[6] + cam.y * K[7] + cam.z * K[8];

	float2 uv;
	uv.x = plane.x / plane.z;
	uv.y = plane.y / plane.z;

	// take nearest neighbour in image
	int2 x_tild;
	x_tild.x = round(uv.x);
	x_tild.y = round(uv.y);

	// check if the current voxel projects into the depth frame
	if (x_tild.x < 0 || x_tild.x >= frame_width || x_tild.y < 0 || x_tild.y >= frame_height || plane.z < 0.0f) {

		return;

	}

	float v = src[x_tild.y * frame_width + x_tild.x];
	float3 pix = (float3)(x_tild.x, x_tild.y, 1.0f);
	float3 pix_k_inv;
	pix_k_inv.x = dot ((float3)(K_inv[0], K_inv[1], K_inv[2]), pix);
	pix_k_inv.y = dot ((float3)(K_inv[3], K_inv[4], K_inv[5]), pix);
	pix_k_inv.z = dot ((float3)(K_inv[6], K_inv[7], K_inv[8]), pix);
	pix_k_inv *= v;

	float to_measurement = sqrt(dot(pix_k_inv, pix_k_inv));
	float to_voxel = sqrt(dot(cam, cam));

	float sdf = to_measurement - to_voxel;

	if (sdf >= -mu) {

		float tsdf = fmin(1.0f, sdf/mu);

		if (isnan(tsdf)) return;

		float prev_tsdf = dest[idx];

		if (isnan(prev_tsdf)) prev_tsdf = 0;

		uint prev_weight = weight[idx];

		// if we haven't had a valid measurement, 
		// then the prev_tsdf is NAN and prev_weight = 0
		uint new_weight = min(MAX_WEIGHT, prev_weight + 1);
		weight[idx] = new_weight;

		dest[idx] = (prev_tsdf * prev_weight + tsdf * new_weight) / (prev_weight + new_weight);
 
	}
}

