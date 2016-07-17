int is_finite(float3 v) {

	return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);

}


kernel void correspondences(
	__global float * v,	//	0
	__global float * n,	//	1
	__global float * pv,	//	2
	__global float * pn,	//	3
	const unsigned int width,	//	4
	const unsigned int height,	//	5
	__global float * t_frame_frame,	//	6
	__global float * t_z,	//	7
	float epsilon_d,	//	8
	float epsilon_theta,	//	9
	__global float * k,	//	10
	__global float * corr_pv,	//	11
	__global float * corr_pn	//	12
) {

	size_t x=get_global_id(0);
	size_t y=get_global_id(1);
	size_t idx=(y*width)+x;

	float3 curr_v=vload3(idx,v);
	float4 curr_v_homo=(float4)(curr_v,1);
	
	float4 persp;
	persp.x=
		(t_frame_frame[0]*curr_v_homo.x)+
		(t_frame_frame[1]*curr_v_homo.y)+
		(t_frame_frame[2]*curr_v_homo.z)+
		(t_frame_frame[3]*curr_v_homo.w);
	persp.y=
		(t_frame_frame[4]*curr_v_homo.x)+
		(t_frame_frame[5]*curr_v_homo.y)+
		(t_frame_frame[6]*curr_v_homo.z)+
		(t_frame_frame[7]*curr_v_homo.w);
	persp.z=
		(t_frame_frame[8]*curr_v_homo.x)+
		(t_frame_frame[9]*curr_v_homo.y)+
		(t_frame_frame[10]*curr_v_homo.z)+
		(t_frame_frame[11]*curr_v_homo.w);
	persp.w=
		(t_frame_frame[12]*curr_v_homo.x)+
		(t_frame_frame[13]*curr_v_homo.y)+
		(t_frame_frame[14]*curr_v_homo.z)+
		(t_frame_frame[15]*curr_v_homo.w);
	
	float3 persp_div=(float3)(curr_v_homo.x/curr_v_homo.w,curr_v_homo.y/curr_v_homo.w,curr_v_homo.z/curr_v_homo.w);

	float3 image_plane;
	image_plane.x=
		(k[0]*persp_div.x)+
		(k[1]*persp_div.y)+
		(k[2]*persp_div.z);
	image_plane.y=
		(k[3]*persp_div.x)+
		(k[4]*persp_div.y)+
		(k[5]*persp_div.z);
	image_plane.z=
		(k[6]*persp_div.x)+
		(k[7]*persp_div.y)+
		(k[8]*persp_div.z);
	
	uint2 u=(uint2)(round(image_plane.x/image_plane.z),round(image_plane.y/image_plane.z));

	size_t lin_idx=u.x+u.y*width;

	float3 nullv=(float3)(0,0,0);
	if (lin_idx>=(width*height)) {

		vstore3(nullv,idx,corr_pn);
		return;

	}

	float3 curr_n=vload3(idx,n);
	float3 curr_pv=vload3(lin_idx,pv);
	float3 curr_pn=vload3(lin_idx,pn);
	if (!(is_finite(curr_v) && is_finite(curr_n) && is_finite(curr_pv) && is_finite(curr_pn))) {

		vstore3(nullv,idx,corr_pn);
		return;

	}

	float4 curr_pv_homo=(float4)(curr_pv,1);
	float4 t_z_curr_v_homo;
	t_z_curr_v_homo.x=
		(t_z[0]*curr_v_homo.x)+
		(t_z[1]*curr_v_homo.y)+
		(t_z[2]*curr_v_homo.z)+
		(t_z[3]*curr_v_homo.w);
	t_z_curr_v_homo.y=
		(t_z[4]*curr_v_homo.x)+
		(t_z[5]*curr_v_homo.y)+
		(t_z[6]*curr_v_homo.z)+
		(t_z[7]*curr_v_homo.w);
	t_z_curr_v_homo.z=
		(t_z[8]*curr_v_homo.x)+
		(t_z[9]*curr_v_homo.y)+
		(t_z[10]*curr_v_homo.z)+
		(t_z[11]*curr_v_homo.w);
	t_z_curr_v_homo.w=
		(t_z[12]*curr_v_homo.x)+
		(t_z[13]*curr_v_homo.y)+
		(t_z[14]*curr_v_homo.z)+
		(t_z[15]*curr_v_homo.w);
	if (dot(t_z_curr_v_homo,curr_pv_homo)>(epsilon_d*epsilon_d)) {

		vstore3(nullv,idx,corr_pn);
		return;

	}

	float3 r_z_curr_n;
	r_z_curr_n.x=
		(t_z[0]*curr_n.x)+
		(t_z[1]*curr_n.y)+
		(t_z[2]*curr_n.z);
	r_z_curr_n.y=
		(t_z[4]*curr_n.x)+
		(t_z[5]*curr_n.y)+
		(t_z[6]*curr_n.z);
	r_z_curr_n.z=
		(t_z[8]*curr_n.x)+
		(t_z[9]*curr_n.y)+
		(t_z[10]*curr_n.z);
	float3 c=cross(r_z_curr_n,curr_pn);
	if (dot(c,c)>(epsilon_theta*epsilon_theta)) {

		vstore3(nullv,idx,corr_pn);
		return;

	}

	vstore3(curr_pv,idx,corr_pv);
	vstore3(curr_pn,idx,corr_pn);

}


kernel void map(
	__global float * v,	//	0
	__global float * corr_pv,	//	1
	__global float * corr_pn,	//	2
	const unsigned int width,	//	3
	const unsigned int height,	//	4
	__global float * ais,	//	5
	__global float * bis,	//	6
	volatile __global unsigned int * count	//	7
) {

	size_t x=get_global_id(0);
	size_t y=get_global_id(1);
	size_t idx=(y*width)+x;
	size_t bi_idx=idx*6;
	global float * bi=bis+bi_idx;
	size_t ai_idx=bi_idx*6;
	global float * ai=ais+ai_idx;


	//	We check to make sure that this was a valid
	//	correspondence, if not we count it and
	//	bail out
	//
	//	Counting invalid correspondences suggested
	//	by Jordan as a way to minimize atomic adds
	float3 n=vload3(idx,corr_pn);
	if ((n.x==0) && (n.y==0) && (n.z==0)) {

		#pragma unroll
		for (size_t i=0;i<(6*6);++i) ai[i]=0;
		#pragma unroll
		for (size_t i=0;i<6;++i) bi[i]=0;

		atomic_add(count,1);
		return;

	}

	float3 p=vload3(idx,v);
	float3 q=vload3(idx,corr_pv);

	float3 c=cross(p,n);
	float pqn=dot(p-q,n);

	//	First row
	ai[0]=c.x*c.x;
	ai[1]=c.x*c.y;
	ai[2]=c.x*c.z;
	ai[3]=c.x*n.x;
	ai[4]=c.x*n.y;
	ai[5]=c.x*n.z;
	//	Second row
	ai[6]=c.y*c.x;
	ai[7]=c.y*c.y;
	ai[8]=c.y*c.z;
	ai[9]=c.y*n.x;
	ai[10]=c.y*n.y;
	ai[11]=c.y*n.z;
	//	Third row
	ai[12]=c.z*c.x;
	ai[13]=c.z*c.y;
	ai[14]=c.z*c.z;
	ai[15]=c.z*n.x;
	ai[16]=c.z*n.y;
	ai[17]=c.z*n.z;
	//	Fourth row
	ai[18]=n.x*c.x;
	ai[19]=n.x*c.y;
	ai[20]=n.x*c.z;
	ai[21]=n.x*n.x;
	ai[22]=n.x*n.y;
	ai[23]=n.x*n.z;
	//	Fifth row
	ai[24]=n.y*c.x;
	ai[25]=n.y*c.y;
	ai[26]=n.y*c.z;
	ai[27]=n.y*n.x;
	ai[28]=n.y*n.y;
	ai[29]=n.y*n.z;
	//	Sixth row
	ai[30]=n.z*c.x;
	ai[31]=n.z*c.y;
	ai[32]=n.z*c.z;
	ai[33]=n.z*n.x;
	ai[34]=n.z*n.y;
	ai[35]=n.z*n.z;

	pqn*=-1;
	bi[0]=c.x*pqn;
	bi[1]=c.y*pqn;
	bi[2]=c.z*pqn;
	bi[3]=n.x*pqn;
	bi[4]=n.y*pqn;
	bi[5]=n.z*pqn;

}


kernel void reduce_a(
	__global float * ais,	//	0
	__global float * a,	//	1
	const unsigned int length	//	2
) {

	size_t x=get_global_id(0);
	size_t y=get_global_id(1);
	size_t idx=(y*6)+x;

	float sum=0;
	for (size_t i=0;i<length;++i) {

		sum+=ais[idx];
		ais+=6*6;

	}
	a[idx]=sum;

}


kernel void reduce_b(
	__global float * bis,	//	0
	__global float * b,	//	1
	const unsigned int length	//	2
) {

	size_t idx=get_global_id(0);

	float sum=0;
	for (size_t i=0;i<length;++i) {

		sum+=bis[idx];
		bis+=6;

	}
	b[idx]=sum;

}
