#include <boost/compute.hpp>
#include <dynfu/boost_compute_detail_type_name_trait.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/kinect_fusion_opencl_pose_estimation_pipeline_block.hpp>
#include <dynfu/opencl_program_factory.hpp>
#include <dynfu/scope.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>


namespace dynfu {


	static boost::compute::program get_program (opencl_program_factory & pf) {

		return pf("pose_estimation");

	}


	static boost::compute::kernel get_correspondences_kernel (opencl_program_factory & pf) {

		auto p=get_program(pf);
		return boost::compute::kernel(p,"correspondences");

	}


	static boost::compute::kernel get_map_kernel (opencl_program_factory & pf) {

		auto p=get_program(pf);
		return boost::compute::kernel(p,"map");

	}


	static boost::compute::kernel get_reduce_a_kernel (opencl_program_factory & pf) {

		auto p=get_program(pf);
		return boost::compute::kernel(p,"reduce_a");

	}


	static boost::compute::kernel get_reduce_b_kernel (opencl_program_factory & pf) {

		auto p=get_program(pf);
		return boost::compute::kernel(p,"reduce_b");

	}


	constexpr std::size_t sizeof_b=6U*sizeof(float);
	constexpr std::size_t sizeof_a=sizeof_b*6U;


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
			corr_(get_correspondences_kernel(pf)),
			map_(get_map_kernel(pf)),
			reduce_a_(get_reduce_a_kernel(pf)),
			reduce_b_(get_reduce_b_kernel(pf)),
			t_frame_frame_(q_.get_context(),sizeof(Eigen::Matrix4f),CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY),
			t_z_(q_.get_context(),sizeof(Eigen::Matrix4f),CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY),
			corr_pv_(frame_width*frame_height,q_.get_context()),
			corr_pn_(frame_width*frame_height,q_.get_context()),
			ais_(q_.get_context(),frame_width*frame_height*sizeof_a),
			bis_(q_.get_context(),frame_width*frame_height*sizeof_b),
			a_(q_.get_context(),sizeof_a,CL_MEM_WRITE_ONLY|CL_MEM_HOST_READ_ONLY),
			b_(q_.get_context(),sizeof_b,CL_MEM_WRITE_ONLY|CL_MEM_HOST_READ_ONLY),
			count_(q_.get_context(),sizeof(std::uint32_t)),
			k_(q_.get_context(),sizeof(Eigen::Matrix3f),CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY),
			v_e_(q_),
			n_e_(q_),
			pv_e_(q_),
			pn_e_(q_),
			epsilon_d_(epsilon_d),
			epsilon_theta_(epsilon_theta),
			frame_width_(frame_width),
			frame_height_(frame_height),
			t_gk_initial_(std::move(t_gk_initial)),
			numit_(numit)
	{

		if (numit_==0) throw std::logic_error("Must iterate at least once");

		//	Correspondences arguments
		corr_.set_arg(4,std::uint32_t(frame_width_));
		corr_.set_arg(5,std::uint32_t(frame_height_));
		corr_.set_arg(6,t_frame_frame_);
		corr_.set_arg(7,t_z_);
		corr_.set_arg(8,epsilon_d_);
		corr_.set_arg(9,epsilon_theta_);
		corr_.set_arg(10,k_);
		corr_.set_arg(11,corr_pv_);
		corr_.set_arg(12,corr_pn_);
		corr_.set_arg(13,count_);

		//	Map arguments
		map_.set_arg(1,corr_pv_);
		map_.set_arg(2,corr_pn_);
		map_.set_arg(3,std::uint32_t(frame_width_));
		map_.set_arg(4,std::uint32_t(frame_height_));
		map_.set_arg(5,ais_);
		map_.set_arg(6,bis_);

		//	Reduce arguments
		std::uint32_t length(frame_height_*frame_width_);
		reduce_a_.set_arg(0,ais_);
		reduce_a_.set_arg(1,a_);
		reduce_a_.set_arg(2,length);
		reduce_b_.set_arg(0,bis_);
		reduce_b_.set_arg(1,b_);
		reduce_b_.set_arg(2,length);

	}


	kinect_fusion_opencl_pose_estimation_pipeline_block::value_type kinect_fusion_opencl_pose_estimation_pipeline_block::operator () (
		measurement_pipeline_block::vertex_value_type::element_type & v,
		measurement_pipeline_block::normal_value_type::element_type & n,
		measurement_pipeline_block::vertex_value_type::element_type * prev_v,
		measurement_pipeline_block::normal_value_type::element_type * prev_n,
		Eigen::Matrix3f k,
		value_type t_gk_minus_one
	) {

		using pv_type=cpu_pipeline_value<value_type::element_type::type>;
	
		if (!(t_gk_minus_one && prev_v && prev_n)) {

			t_gk_minus_one=std::make_unique<pv_type>();
			static_cast<pv_type &>(*t_gk_minus_one).emplace(t_gk_initial_);

			return t_gk_minus_one;

		}

		//	Get input vectors and bind parameters
		auto && vb=v_e_(v);
		corr_.set_arg(0,vb);
		auto && nb=n_e_(n);
		corr_.set_arg(1,nb);
		auto && pvb=pv_e_(*prev_v);
		corr_.set_arg(2,pvb);
		auto && pnb=pn_e_(*prev_n);
		corr_.set_arg(3,pnb);

		Eigen::Matrix4f t_frame_frame(Eigen::Matrix4f::Identity());
		Eigen::Matrix4f t_z(t_gk_minus_one->get());

		k.transposeInPlace();
		auto kw=q_.enqueue_write_buffer_async(k_,0,sizeof(k),&k);
		auto kwg=make_scope_exit([&] () noexcept {	kw.wait();	});

		//	Allow up to 90% of points to be rejected
		std::uint32_t threshold(frame_height_*frame_width_);
		threshold/=10U;
		threshold*=9U;

		for (std::size_t i=0;i<numit_;++i) {

			//	Upload matrices to the GPU
			t_frame_frame.transposeInPlace();
			auto tffw=q_.enqueue_write_buffer_async(t_frame_frame_,0,sizeof(t_frame_frame),&t_frame_frame);
			auto tffwg=make_scope_exit([&] () noexcept {	tffw.wait();	});
			t_z.transposeInPlace();
			auto tzw=q_.enqueue_write_buffer_async(t_z_,0,sizeof(t_z),&t_z);
			auto tzwg=make_scope_exit([&] () noexcept {	tzw.wait();	});

			//	Enqueue correspondences kernel
			std::uint32_t count(0);
			auto cw=q_.enqueue_write_buffer_async(count_,0,sizeof(count),&count);
			auto cwg=make_scope_exit([&] () noexcept {	cw.wait();	});
			std::size_t extent []={frame_width_,frame_height_};
			q_.enqueue_nd_range_kernel(corr_,2,nullptr,extent,nullptr);
			auto cr=q_.enqueue_read_buffer_async(count_,0,sizeof(count),&count);
			auto crg=make_scope_exit([&] () noexcept {	cr.wait();	});

			//	Map
			map_.set_arg(0,vb);
			q_.enqueue_nd_range_kernel(map_,2,nullptr,extent,nullptr);

			//	Reduce
			std::size_t a_extent []={6,6};
			q_.enqueue_nd_range_kernel(reduce_a_,2,nullptr,a_extent,nullptr);
			//	Just use first element of a_extent for 1D range on
			//	reducing b
			q_.enqueue_nd_range_kernel(reduce_b_,1,nullptr,a_extent,nullptr);

			//	Download results
			Eigen::Matrix<float,6,6> a;
			auto ar=q_.enqueue_read_buffer_async(a_,0,sizeof(a),&a);
			auto arg=make_scope_exit([&] () noexcept {	ar.wait();	});
			Eigen::Matrix<float,6,1> b;
			auto br=q_.enqueue_read_buffer_async(b_,0,sizeof(b),&b);
			auto brg=make_scope_exit([&] () noexcept {	br.wait();	});

			//	Make sure there were enough correspondences
			cr.wait();
			crg.release();

			std::cout << "Count: " << count << std::endl;

			ar.wait();
			arg.release();
			br.wait();
			brg.release();

			//	Solve system
			Eigen::Matrix<float,6,1> x=a.colPivHouseholderQr().solve(b);

			float alpha=x(0);
			float beta=x(1);
			float gamma=x(2);
			float tx=x(3);
			float ty=x(4);
			float tz=x(5);

			t_z <<  1.0f, -gamma, beta, tx,
					 gamma, 1.0f, -alpha, ty,
				 	 -beta, alpha, 1.0f, tz,
					 0.0f, 0.0f, 0.0f, 1.0f;
			t_frame_frame=t_gk_minus_one->get().inverse() * t_z;

			std::cout << "T_z " << std::endl << t_z << std::endl;

		}



		auto && pv=dynamic_cast<pv_type &>(*t_gk_minus_one);
		pv.emplace(t_z);

		return t_gk_minus_one;
	
	}


}
