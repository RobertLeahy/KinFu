
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
 *	depth_width - width of depth image (for indexing)
 *
 */
kernel void tsdf_kernel(__global float * src, __global float* dest, 
 const unsigned int tsdf_width, const unsigned int tsdf_height, const unsigned int tsdf_depth,
 __global const float* proj_view, __global const float* K, __global const float* K_inv, 
 __global const float* t_gk, const float mu, const unsigned int frame_width, const unsigned int frame_height) {

	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	unsigned int z = get_global_id(2);

	// From http://stackoverflow.com/questions/10903149/how-do-i-compute-the-linear-index-of-a-3d-coordinate-and-vice-versa
	unsigned int idx = x + (tsdf_width+1)*y + ((tsdf_width + 1) * (tsdf_height + 1))*z;

	// we can't compute at these bounds?
	if (x >= tsdf_width || y >= tsdf_height || z >= tsdf_depth ) {
		 dest[idx] = NAN; // TODO: what case do we have here?
		 return;
	} 	

	// x,y,z give the voxel id, but we need to compute the position in 3D space.

	// 2.0^3 m volume
	float4 element_size = (float4)(x/2.0, y/2.0, z/2.0, 1);
	float4 p = (float4)(x,y,z,1) * element_size;

	// do projection view stuff (proj_view has projection matrix and the sensor pose estimation invesere (view))
	float4 x_dot_homo;
	x_dot_homo.x = proj_view[0]*p.x + proj_view[1]*p.y + proj_view[2]*p.z + proj_view[3]*p.w;
	x_dot_homo.y = proj_view[4]*p.x + proj_view[5]*p.y + proj_view[6]*p.z + proj_view[7]*p.w;
	x_dot_homo.z = proj_view[8]*p.x + proj_view[9]*p.y + proj_view[10]*p.z + proj_view[11]*p.w;
	x_dot_homo.w = proj_view[12]*p.x + proj_view[13]*p.y + proj_view[14]*p.z + proj_view[15]*p.w;

	// perspective division
	float3 x_dot_glob = (float3)(x_dot_homo.x/x_dot_homo.w, x_dot_homo.y/x_dot_homo.w, x_dot_homo.z/x_dot_homo.w);
	
	// multiply by camera matrix
	float3 x_dot;
	x_dot.x = K[0]*x_dot_glob.x + K[1]*x_dot_glob.y + K[2]*x_dot_glob.z;
	x_dot.y = K[3]*x_dot_glob.x + K[4]*x_dot_glob.y + K[5]*x_dot_glob.z;
	x_dot.z = K[6]*x_dot_glob.x + K[7]*x_dot_glob.y + K[8]*x_dot_glob.z;

	// dehomogenize and take nearest neighbour
	int2 x_tild;
	x_tild.x = round(x_dot.x/x_dot.z);
	x_tild.y = round(x_dot.y/x_dot.z);
	
	// check if the current voxel projects into the depth frame
	if (x_tild.x < 0 || x_tild.x > frame_width || x_tild.y < 0 || x_tild.y > frame_height) {
		// current voxel doesn't project into our frame of reference
		dest[idx] = NAN; // TODO: what case do we have here?
		return;
	}
	
	float3 K_inv_x;
	K_inv_x.x = K_inv[0]*x_tild.x + K_inv[1]*x_tild.y + K_inv[2];
	K_inv_x.y = K_inv[3]*x_tild.x + K_inv[4]*x_tild.y + K_inv[5];
	K_inv_x.z = K_inv[6]*x_tild.x + K_inv[7]*x_tild.y + K_inv[8];
	
	float lambda = l2_norm(K_inv_x);
	
	// R_k(x)
	float R_k = src[x_tild.y * frame_width + x_tild.x];
	
	// t_gk - p
	float3 t_gk_p = (float3)(t_gk[0] - x, t_gk[1] - y, t_gk[2] - z);
	
	float nu = (1.0/lambda) * l2_norm(t_gk_p) - R_k;
	
	float sign = 1;
	if (signbit(nu)) sign = -1;
	
	if (nu >= -mu) {
		dest[idx] = min(1.0f, nu/mu) * sign;
	} else {
		dest[idx] = NAN; // TODO: what case do we have here?
	}
	
}

