#include <boost/compute.hpp>
#include <dynfu/boost_compute_detail_type_name_trait.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/kinect_fusion_opencl_pose_estimation_pipeline_block.hpp>
#include <dynfu/opencl_program_factory.hpp>
#include <dynfu/optional.hpp>
#include <dynfu/scope.hpp>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>


namespace dynfu {


	constexpr std::size_t sizeof_mats=sizeof(float)*(21U+6U);


	kinect_fusion_opencl_pose_estimation_pipeline_block::kinect_fusion_opencl_pose_estimation_pipeline_block (
		opencl_program_factory & pf,
		boost::compute::command_queue q,
		float epsilon_d,
		float epsilon_theta,
		std::size_t frame_width,
		std::size_t frame_height,
		Eigen::Matrix4f t_gk_initial,
		std::size_t numit
	)	:	q_(std::move(q)),
			t_z_(q_.get_context(),sizeof(Eigen::Matrix4f),CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY),
			t_gk_prev_inverse_(q_.get_context(),sizeof(Eigen::Matrix4f),CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY),
			k_(q_.get_context(),sizeof(Eigen::Matrix3f),CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY),
			mats_(q_.get_context(),sizeof_mats*frame_height*frame_width),
			e_(q_),
			p_e_(q_),
			epsilon_d_(epsilon_d),
			epsilon_theta_(epsilon_theta),
			frame_width_(frame_width),
			frame_height_(frame_height),
			t_gk_initial_(std::move(t_gk_initial)),
			numit_(numit)
	{

		if (numit_==0) throw std::logic_error("Must iterate at least once");

		auto p=pf("pose_estimation");
		corr_=p.create_kernel("correspondences");
		sum_=p.create_kernel("sum");

		//	Correspondences arguments
		corr_.set_arg(2,t_gk_prev_inverse_);
		corr_.set_arg(3,t_z_);
		corr_.set_arg(4,epsilon_d_);
		corr_.set_arg(5,epsilon_theta_);
		corr_.set_arg(6,k_);
		corr_.set_arg(7,mats_);

		//	Sum arguments
		sum_.set_arg(0,mats_);

	}


	kinect_fusion_opencl_pose_estimation_pipeline_block::value_type kinect_fusion_opencl_pose_estimation_pipeline_block::operator () (
		measurement_pipeline_block::value_type::element_type & map,
		measurement_pipeline_block::value_type::element_type * prev_map,
		Eigen::Matrix3f k,
		value_type t_gk_minus_one
	) {

		using pv_type=cpu_pipeline_value<value_type::element_type::type>;
	
		if (!(t_gk_minus_one && prev_map)) {

			t_gk_minus_one=std::make_unique<pv_type>();
			static_cast<pv_type &>(*t_gk_minus_one).emplace(t_gk_initial_);

			return t_gk_minus_one;

		}

		//	Get input vectors and bind parameters
		auto && map_b=e_(map);
		corr_.set_arg(0,map_b);
		auto && prev_map_b=p_e_(*prev_map);
		corr_.set_arg(1,prev_map_b);

		auto && pv=dynamic_cast<pv_type &>(*t_gk_minus_one);
		auto t_gk_minus_one_m=pv.get();
		Eigen::Matrix4f t_z(t_gk_minus_one_m);
		Eigen::Matrix4f t_gk_prev_inverse(t_gk_minus_one_m.inverse());

		t_gk_prev_inverse.transposeInPlace();
		auto tgkw=q_.enqueue_write_buffer_async(t_gk_prev_inverse_,0,sizeof(t_gk_prev_inverse),&t_gk_prev_inverse);
		auto tgkwg=make_scope_exit([&] () noexcept {	tgkw.wait();	});

		k.transposeInPlace();
		auto kw=q_.enqueue_write_buffer_async(k_,0,sizeof(k),&k);
		auto kwg=make_scope_exit([&] () noexcept {	kw.wait();	});

		for (std::size_t i=0;i<numit_;++i) {

			//	Upload current estimate to the GPU
			t_z.transposeInPlace();
			auto tzw=q_.enqueue_write_buffer_async(t_z_,0,sizeof(t_z),&t_z);
			auto tzwg=make_scope_exit([&] () noexcept {	tzw.wait();	});

			//	Enqueue correspondences kernel
			std::size_t extent []={frame_width_,frame_height_};
			q_.enqueue_nd_range_kernel(corr_,2,nullptr,extent,nullptr);

			//	Sum
			for (std::size_t prev_length=frame_height_*frame_width_,length=prev_length/2U,mul=2U;length!=0;prev_length=length,length/=2U,mul*=2U) {

				sum_.set_arg(1,std::uint32_t(mul));
				sum_.set_arg(2,std::int32_t((prev_length%2U)!=0));
				q_.enqueue_nd_range_kernel(sum_,1,nullptr,&length,nullptr);

			}

			//	Collect results
			float buffer [21U+6U];
			q_.enqueue_read_buffer(mats_,0,sizeof(buffer),buffer);
			using a_type=Eigen::Matrix<float,6,6>;
			a_type a;
			a <<	buffer[0],buffer[1],buffer[2],buffer[3],buffer[4],buffer[5],
					buffer[1],buffer[6],buffer[7],buffer[8],buffer[9],buffer[10],
					buffer[2],buffer[7],buffer[11],buffer[12],buffer[13],buffer[14],
					buffer[3],buffer[8],buffer[12],buffer[15],buffer[16],buffer[17],
					buffer[4],buffer[9],buffer[13],buffer[16],buffer[18],buffer[19],
					buffer[5],buffer[10],buffer[14],buffer[17],buffer[19],buffer[20];
			using b_type=Eigen::Matrix<float,6,1>;
			b_type b;
			std::memcpy(&b,buffer+21U,sizeof(b));

			//	Solve system
			Eigen::FullPivHouseholderQR<a_type> solver(a);
			b_type x=solver.solve(b);

			float alpha=x(0);
			float beta=x(1);
			float gamma=x(2);
			float tx=x(3);
			float ty=x(4);
			float tz=x(5);

			Eigen::Matrix4f t_inc;
			t_inc <<  1.0f, -gamma, beta, tx,
					 gamma, 1.0f, -alpha, ty,
				 	 -beta, alpha, 1.0f, tz,
					 0.0f, 0.0f, 0.0f, 1.0f;
			t_z.transposeInPlace();
			t_z=t_inc*t_z;

		}

		pv.emplace(t_z);

		return t_gk_minus_one;
	
	}


}
