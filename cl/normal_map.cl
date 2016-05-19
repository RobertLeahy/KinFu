kernel void normal_map(__global float * src, __global float* dest, 
 const unsigned int width, const unsigned int height) {

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

	// we can't compute at these bounds?
	if (x >= width-1 || y >= height-1) {
		 vstore3((float3)(0,0,0), y*width + x, dest);
		 return;
	} 	

	float3 V_k_u_v = vload3(y*width + x, src);
	float3 V_k_u1_v = vload3(y*width + x + 1, src);
	float3 V_k_u_v1 = vload3((y+1)*width + x, src);
	
	float3 u = V_k_u1_v - V_k_u_v;
	float3 v = V_k_u_v1 - V_k_u_v;
	
	float3 x_vec;
	
	// compute V_k_u cross V_k_v
	x_vec.x = u.y*v.z - u.z*v.y; 
	x_vec.y = u.z*v.x - u.x*v.z;
	x_vec.z = u.x*v.y - u.y*v.x;
	
	float l2_norm_x = sqrt(x_vec.x*x_vec.x + x_vec.y*x_vec.y + x_vec.z*x_vec.z);
	
	vstore3(x_vec/l2_norm_x, y*width + x, dest);

}