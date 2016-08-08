#define SIZEOF_MATS (27U)


float3 matrixmul3(const __constant float * m, float3 v) {

    float3 retr;
    retr.x=dot(vload3(0,m),v);
    retr.y=dot(vload3(1,m),v);
    retr.z=dot(vload3(2,m),v);

    return retr;

}


float4 matrixmul4(const __constant float * m, float4 v) {

    float4 retr;
    retr.x=dot(vload4(0,m),v);
    retr.y=dot(vload4(1,m),v);
    retr.z=dot(vload4(2,m),v);
    retr.w=dot(vload4(3,m),v);

    return retr;

}


int is_finite(float3 v) {

	return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);

}


void zero_ab (__local float * ms) {

	#pragma unroll
	for (size_t i=0;i<SIZEOF_MATS;++i) ms[i]=0;

}


void icp (float3 p, float3 q, float3 n, __local float * ms) {

	float3 c=cross(p,n);
	float pqn=dot(p-q,n);

	//	First row
	ms[0]=c.x*c.x;
	ms[1]=c.x*c.y;
	ms[2]=c.x*c.z;
	ms[3]=c.x*n.x;
	ms[4]=c.x*n.y;
	ms[5]=c.x*n.z;
	//	Second row
	ms[6]=c.y*c.y;
	ms[7]=c.y*c.z;
	ms[8]=c.y*n.x;
	ms[9]=c.y*n.y;
	ms[10]=c.y*n.z;
	//	Third row
	ms[11]=c.z*c.z;
	ms[12]=c.z*n.x;
	ms[13]=c.z*n.y;
	ms[14]=c.z*n.z;
	//	Fourth row
	ms[15]=n.x*n.x;
	ms[16]=n.x*n.y;
	ms[17]=n.x*n.z;
	//	Fifth row
	ms[18]=n.y*n.y;
	ms[19]=n.y*n.z;
	//	Sixth row
	ms[20]=n.z*n.z;

	pqn*=-1;
	ms[21]=c.x*pqn;
	ms[22]=c.y*pqn;
	ms[23]=c.z*pqn;
	ms[24]=n.x*pqn;
	ms[25]=n.y*pqn;
	ms[26]=n.z*pqn;

}


void correspondences_impl(
	const __global float * map,
	const __global float * prev_map,
	const __constant float * t_gk_prev_inverse,
	const __constant float * t_z,
	float epsilon_d,
	float epsilon_theta,
	const __constant float * k,
	size_t x,
	size_t y,
	size_t width,
	size_t height,
	__local float * ms
) {

	size_t idx=(y*width)+x;
	idx*=2U;

	float3 curr_pv=vload3(idx,prev_map);
	if (!is_finite(curr_pv)) {

		zero_ab(ms);
		return;

	}

	float4 curr_pv_homo=(float4)(curr_pv,1);

	float4 v_pcp_h=matrixmul4(t_gk_prev_inverse,curr_pv_homo);

	float3 v_pcp=(float3)(v_pcp_h.x, v_pcp_h.y, v_pcp_h.z);
	float3 uv3=matrixmul3(k,v_pcp);

	int2 u=(int2)(round(uv3.x/uv3.z),round(uv3.y/uv3.z));

	if ((u.x<0) || (u.y<0) || (((size_t)u.x)>=width) || (((size_t)u.y)>=height)) {

		zero_ab(ms);
		return;

	}

	size_t lin_idx=u.x+u.y*width;

	float3 curr_pn=vload3(idx+1U,prev_map);
	//	These are in current camera space
	lin_idx*=2U;
	float3 curr_v=vload3(lin_idx,map);
	float3 curr_n=vload3(lin_idx+1U,map);
	//	TODO: Should we be checking the current normal?
	if (!(is_finite(curr_v) && is_finite(curr_pn))) {

		zero_ab(ms);
		return;

	}

	float4 curr_v_homo=(float4)(curr_v,1);
	float4 t_z_curr_v_homo=matrixmul4(t_z,curr_v_homo);
	float4 t_z_curr_v_homo_curr_pv_homo=t_z_curr_v_homo-curr_pv_homo;
	float3 t_z_curr_v=(float3)(t_z_curr_v_homo.x,t_z_curr_v_homo.y,t_z_curr_v_homo.z);
	if (dot(t_z_curr_v_homo_curr_pv_homo,t_z_curr_v_homo_curr_pv_homo)>(epsilon_d*epsilon_d)) {

		zero_ab(ms);
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
	float3 crzcncpn=cross(r_z_curr_n,curr_pn);
	if (dot(crzcncpn,crzcncpn)>(epsilon_theta*epsilon_theta)) {

		zero_ab(ms);
		return;

	}

	icp(t_z_curr_v,curr_pv,curr_pn,ms);

}


void parallel_sum_impl(
	size_t l,
	size_t size,
	__global float * dest,
	__local float * lmats
) {

	for (size_t stride=size/2U;stride>0;stride/=2U) {

		//	Wait for everything else to finish
		barrier(CLK_LOCAL_MEM_FENCE);

		//	Perform add if needed
		if (l>=stride) continue;

		local float * a=lmats+(l*SIZEOF_MATS);
		const local float * b=lmats+((l+stride)*SIZEOF_MATS);
		#pragma unroll
		for (size_t i=0;i<SIZEOF_MATS;++i) a[i]+=b[i];

	}

	//	Write out
	if (l!=0) return;
	#pragma unroll
	for (size_t i=0;i<SIZEOF_MATS;++i,++dest) *dest=lmats[i];

}


kernel void correspondences(
	const __global float * map,	//	0
	const __global float * prev_map,	//	1
	const __constant float * t_gk_prev_inverse,	//	2
	const __constant float * t_z,	//	3
	float epsilon_d,	//	4
	float epsilon_theta,	//	5
	const __constant float * k,	//	6
	unsigned int width,	//	7
	unsigned int height,	//	8
	__global float * mats,	//	9
	__local float * lmats	//	10
) {

	size_t idx=get_global_id(0);
	size_t x=idx%width;
	size_t y=idx/width;
	size_t l=get_local_id(0);

	correspondences_impl(
		map,
		prev_map,
		t_gk_prev_inverse,
		t_z,
		epsilon_d,
		epsilon_theta,
		k,
		x,
		y,
		width,
		height,
		lmats+(SIZEOF_MATS*l)
	);

	parallel_sum_impl(
		l,
		get_local_size(0),
		mats+(get_group_id(0)*SIZEOF_MATS),
		lmats
	);

}


//	ASSUMPTION:
//
//	Local group size IS NOT ODD!!!!
kernel void parallel_sum(
	const __global float * src,	//	0
	__global float * dest,	//	1
	__local float * lmats	//	2
) {

	//	Get identifiers
	size_t g=get_global_id(0);
	size_t l=get_local_id(0);

	//	Load into local memory
	const global float * ld=src+(g*SIZEOF_MATS);
	local float * sv=lmats+(l*SIZEOF_MATS);
	#pragma unroll
	for (size_t i=0;i<SIZEOF_MATS;++i,++sv,++ld) *sv=*ld;

	parallel_sum_impl(
		l,
		get_local_size(0),
		dest+(get_group_id(0)*SIZEOF_MATS),
		lmats
	);

}


kernel void serial_sum(
	__global float * mats,	//	0
	unsigned int length	//	1
) {

	size_t i=get_global_id(0);
	float sum=0;

	for (size_t j=0;j<length;++j) sum+=mats[(j*SIZEOF_MATS)+i];

	mats[i]=sum;

}
