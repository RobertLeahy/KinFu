float3 matrixmul3(const __global float * m, float3 v) {

    float3 retr;
    retr.x=dot(vload3(0,m),v);
    retr.y=dot(vload3(1,m),v);
    retr.z=dot(vload3(2,m),v);

    return retr;

}


float4 matrixmul4(const __global float * m, float4 v) {

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


struct __attribute__((packed)) kernel_data {

	float3 v;
	float3 pv;
	float3 pn;
	int valid;
	float a [36];
	float b [6];

};


kernel void load_p(
	__global struct kernel_data * data,	//	0
	const __global float * v,	//	1
	const __global float * n	//	2
) {

	size_t x=get_global_id(0);
	size_t width=get_global_size(0);
	size_t y=get_global_id(1);
	size_t idx=(y*width)+x;

	global struct kernel_data * curr=data+idx;
	curr->pv=vload3(idx,v);
	curr->pn=vload3(idx,n);
	
}


struct __attribute__((packed)) vertex_and_normal {

	float3 v;
	float3 n;

};


kernel void load_vn(
	__global struct vertex_and_normal * vn,	//	0
	const __global float * v,	//	1
	const __global float * n	//	2
) {

	size_t x=get_global_id(0);
	size_t width=get_global_size(0);
	size_t y=get_global_id(1);
	size_t idx=(y*width)+x;

	global struct vertex_and_normal * curr=vn+idx;
	curr->v=vload3(idx,v);
	curr->n=vload3(idx,n);

}


void zero_ab (__global float * a, __global float * b) {

	#pragma unroll
	for (size_t i=0;i<(6*6);++i) a[i]=0;
	#pragma unroll
	for (size_t i=0;i<6;++i) b[i]=0;

}


void icp (float3 p, float3 q, float3 n, __global float * a, __global float * b) {

	float3 c=cross(p,n);
	float pqn=dot(p-q,n);

	//	First row
	a[0]=c.x*c.x;
	a[1]=c.x*c.y;
	a[2]=c.x*c.z;
	a[3]=c.x*n.x;
	a[4]=c.x*n.y;
	a[5]=c.x*n.z;
	//	Second row
	a[6]=c.y*c.x;
	a[7]=c.y*c.y;
	a[8]=c.y*c.z;
	a[9]=c.y*n.x;
	a[10]=c.y*n.y;
	a[11]=c.y*n.z;
	//	Third row
	a[12]=c.z*c.x;
	a[13]=c.z*c.y;
	a[14]=c.z*c.z;
	a[15]=c.z*n.x;
	a[16]=c.z*n.y;
	a[17]=c.z*n.z;
	//	Fourth row
	a[18]=n.x*c.x;
	a[19]=n.x*c.y;
	a[20]=n.x*c.z;
	a[21]=n.x*n.x;
	a[22]=n.x*n.y;
	a[23]=n.x*n.z;
	//	Fifth row
	a[24]=n.y*c.x;
	a[25]=n.y*c.y;
	a[26]=n.y*c.z;
	a[27]=n.y*n.x;
	a[28]=n.y*n.y;
	a[29]=n.y*n.z;
	//	Sixth row
	a[30]=n.z*c.x;
	a[31]=n.z*c.y;
	a[32]=n.z*c.z;
	a[33]=n.z*n.x;
	a[34]=n.z*n.y;
	a[35]=n.z*n.z;

	pqn*=-1;
	b[0]=c.x*pqn;
	b[1]=c.y*pqn;
	b[2]=c.z*pqn;
	b[3]=n.x*pqn;
	b[4]=n.y*pqn;
	b[5]=n.z*pqn;

}


kernel void correspondences(
	__global struct kernel_data * data,	//	0
	const __global struct vertex_and_normal * vn,	//	1
	const __global float * t_gk_prev_inverse,	//	2
	const __global float * t_z,	//	3
	float epsilon_d,	//	4
	float epsilon_theta,	//	5
	const __global float * k,	//	6
	__global float * corr_v,	//	7
	__global float * corr_pn,	//	8
	__global float * ais,	//	9
	__global float * bis	//	10
) {

	size_t x=get_global_id(0);
	size_t width=get_global_size(0);
	size_t y=get_global_id(1);
	size_t idx=(y*width)+x;
	global struct kernel_data * curr=data+idx;
	global float * a=ais+(idx*6U*6U);
	global float * b=bis+(idx*6U);

	float3 curr_pv=curr->pv;
	float4 curr_pv_homo=(float4)(curr_pv,1);

	float4 v_pcp_h=matrixmul4(t_gk_prev_inverse,curr_pv_homo);

	float3 v_pcp=(float3)(v_pcp_h.x, v_pcp_h.y, v_pcp_h.z);
	float3 uv3=matrixmul3(k,v_pcp);

	int2 u=(int2)(round(uv3.x/uv3.z),round(uv3.y/uv3.z));

	int lin_idx=u.x+u.y*width;

	if (lin_idx>=(width*get_global_size(1)) || lin_idx<0) {

		zero_ab(a,b);
		return;

	}

	float3 curr_pn=curr->pn;
	//	These are in current camera space
	global struct vertex_and_normal * curr_vn=vn+idx;
	float3 curr_n=curr_vn->n;
	float3 curr_v=curr_vn->v;
	//	TODO: Should we be checking the current normal?
	if (!(is_finite(curr_v) && is_finite(curr_pv) && is_finite(curr_pn))) {

		zero_ab(a,b);
		return;

	}

	float4 curr_v_homo=(float4)(curr_v,1);
	float4 t_z_curr_v_homo=matrixmul4(t_z,curr_v_homo);
	float4 t_z_curr_v_homo_curr_pv_homo=t_z_curr_v_homo-curr_pv_homo;
	float3 t_z_curr_v=(float3)(t_z_curr_v_homo.x,t_z_curr_v_homo.y,t_z_curr_v_homo.z);
	if (dot(t_z_curr_v_homo_curr_pv_homo,t_z_curr_v_homo_curr_pv_homo)>(epsilon_d*epsilon_d)) {

		zero_ab(a,b);
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

		zero_ab(a,b);
		return;

	}

	icp(t_z_curr_v,curr_pv,curr_pn,a,b);

}


kernel void reduce_a(
	const __global float * ais,	//	0
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
	const __global float * bis,	//	0
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
