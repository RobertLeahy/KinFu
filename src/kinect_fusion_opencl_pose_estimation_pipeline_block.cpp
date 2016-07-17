#include <boost/compute.hpp>
#include <dynfu/boost_compute_detail_type_name_trait.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/kinect_fusion_opencl_pose_estimation_pipeline_block.hpp>
#include <dynfu/opencl_program_factory.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
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
			t_frame_frame_(q_.get_context(),sizeof(Eigen::Matrix4f)),
			t_z_(q_.get_context(),sizeof(Eigen::Matrix4f)),
			corr_pv_(frame_width*frame_height,q_.get_context()),
			corr_pn_(frame_width*frame_height,q_.get_context()),
			ais_(q_.get_context(),frame_width*frame_height*sizeof_a),
			bis_(q_.get_context(),frame_width*frame_height*sizeof_b),
			a_(q_.get_context(),sizeof_a),
			b_(q_.get_context(),sizeof_b),
			count_(q_.get_context(),sizeof(std::uint32_t)),
			k_(q_.get_context(),sizeof(Eigen::Matrix3f)),
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

			auto ptr=std::make_unique<pv_type>();
			ptr->emplace(t_gk_initial_);

			return ptr;

		}

		//	Get input vectors and bind parameters
		opencl_vector_pipeline_value_extractor<measurement_pipeline_block::vertex_value_type::element_type::type::value_type> v_e(q_);
		auto && vb=v_e(v);
		corr_.set_arg(0,vb);
		opencl_vector_pipeline_value_extractor<measurement_pipeline_block::normal_value_type::element_type::type::value_type> n_e(q_);
		auto && nb=n_e(n);
		corr_.set_arg(1,nb);
		opencl_vector_pipeline_value_extractor<measurement_pipeline_block::vertex_value_type::element_type::type::value_type> pv_e(q_);
		auto && pvb=pv_e(*prev_v);
		corr_.set_arg(2,pvb);
		opencl_vector_pipeline_value_extractor<measurement_pipeline_block::normal_value_type::element_type::type::value_type> pn_e(q_);
		auto && pnb=pn_e(*prev_n);
		corr_.set_arg(3,pnb);

		Eigen::Matrix4f t_frame_frame(Eigen::Matrix4f::Identity());
		Eigen::Matrix4f t_z(Eigen::Matrix4f::Identity());

		k.transposeInPlace();
		q_.enqueue_write_buffer(k_,0,sizeof(k),&k);

		std::uint32_t threshold(frame_height_*frame_width_);
		threshold/=10U;

		for (std::size_t i=0;i<numit_;++i) {

			//	Upload matrices to the GPU
			t_frame_frame.transposeInPlace();
			q_.enqueue_write_buffer(t_frame_frame_,0,sizeof(t_frame_frame),&t_frame_frame);
			t_z.transposeInPlace();
			q_.enqueue_write_buffer(t_z_,0,sizeof(t_z),&t_z);

			//	Enqueue correspondences kernel
			std::uint32_t count(0);
			q_.enqueue_write_buffer(count_,0,sizeof(count),&count);
			std::size_t extent []={frame_width_,frame_height_};
			q_.enqueue_nd_range_kernel(corr_,2,nullptr,extent,nullptr);
			q_.enqueue_read_buffer(count_,0,sizeof(count),&count);
			if (count>threshold) {

				std::ostringstream ss;
				ss << "Tracking lost: Could not find correspondences for " << count << "/" << (frame_height_*frame_width_) << " points (maximum allowable is " << threshold << ")";
				throw tracking_lost_error(ss.str());

			}

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
			q_.enqueue_read_buffer(a_,0,sizeof(a),&a);
			Eigen::Matrix<float,6,1> b;
			q_.enqueue_read_buffer(b_,0,sizeof(b),&b);

			//	Solve system
			Eigen::Matrix<float,6,1> x=a.colPivHouseholderQr().solve(b);

			t_z << 	1.0f, x(0), -x(2), x(3),
					-x(0), 1.0f, x(1), x(4),
					x(2), -x(1), 1.0f, x(5),
					0.0f, 0.0f, 0.0f, 1.0f;
			t_frame_frame=t_z;

		}

		auto && pv=dynamic_cast<pv_type &>(*t_gk_minus_one);
		auto prev=pv.get();
		pv.emplace(t_z*prev);

		return t_gk_minus_one;
	
	}


}
