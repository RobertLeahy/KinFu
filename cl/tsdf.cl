
#define l2_norm(t) \
	( sqrt(t.x*t.x + t.y*t.y + t.z*t.z) )

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
 const float tsdf_extent_w, const float tsdf_extent_h, const float tsdf_extent_d) {

 	// get the x, y, z of the current voxel from memory
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	unsigned int z = get_global_id(2);

	// Determine the linear index using the x,y,z components
	// From http://stackoverflow.com/questions/10903149/how-do-i-compute-the-linear-index-of-a-3d-coordinate-and-vice-versa
	unsigned int idx = x + (tsdf_width)*y + ((tsdf_width) * (tsdf_height))*z;

	// OOB
	if (x > tsdf_width || y > tsdf_height || z > tsdf_depth ) {
		 dest[idx] = NAN; // TODO: what case do we have here?
		 return;
	} 	

	// x,y,z give the voxel id, but we need to compute the position in 3D space.

	// volume extends from origin to tsdf_extent
	float p_x = ((float)x + 0.5) * tsdf_extent_w/tsdf_width;
	float p_y = ((float)y + 0.5) * tsdf_extent_h/tsdf_height;
	float p_z = ((float)z + 0.5) * tsdf_extent_d/tsdf_depth;
	float4 p = (float4)(p_x, p_y, p_z, 1);

	// project into normalized device coordinates (NDC)
	float4 x_ndc;
	x_ndc.x = proj_view[0]*p.x + proj_view[1]*p.y + proj_view[2]*p.z + proj_view[3]*p.w;
	x_ndc.y = proj_view[4]*p.x + proj_view[5]*p.y + proj_view[6]*p.z + proj_view[7]*p.w;
	x_ndc.z = proj_view[8]*p.x + proj_view[9]*p.y + proj_view[10]*p.z + proj_view[11]*p.w;
	x_ndc.w = proj_view[12]*p.x + proj_view[13]*p.y + proj_view[14]*p.z + proj_view[15]*p.w;

	// perspective division
	float3 x_dot_glob = (float3)(x_ndc.x/x_ndc.w, x_ndc.y/x_ndc.w, x_ndc.z/x_ndc.w);
	
	// convert from NDC to image coords
	float3 x_dot;
	x_dot.x = (x_dot_glob.x + 1.0f) * K[2];
	x_dot.y = (x_dot_glob.y + 1.0f) * K[5]; 

	// take nearest neighbour in image
	int2 x_tild;
	x_tild.x = round(x_dot.x);
	x_tild.y = round(x_dot.y);

	// check if the current voxel projects into the depth frame
	if (x_tild.x < 0 || x_tild.x > frame_width || x_tild.y < 0 || x_tild.y > frame_height || x_ndc.z < 0.0f) {

		dest[idx] = 1.0f;
		return;

	}

	// Compute Eqn (7)
	float3 K_inv_x;
	K_inv_x.x = (x_tild.x - K[2]) / K[0];
	K_inv_x.y = (x_tild.y - K[5]) / K[4];
	K_inv_x.z = 1;
	
	float lambda = l2_norm(K_inv_x);

	// Compute Eqn (6)

	// R_k(x)
	float R_k = src[x_tild.y * frame_width + x_tild.x];
	
	// t_gk - p
	float3 t_gk_p = (float3)(t_gk[0] - p_x, t_gk[1] - p_y, t_gk[2] - p_z);
	
	float nu = (1.0/lambda) * l2_norm(t_gk_p) - R_k;
	
	if (nu >= -mu) {

		dest[idx] = -fmin(1.0f, nu/mu);

	} else {

		dest[idx] = NAN; // Don't integrate if we get here

	}
	
}
