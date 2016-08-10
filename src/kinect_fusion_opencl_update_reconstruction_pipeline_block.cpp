#include <dynfu/boost_compute_detail_type_name_trait.hpp>
#include <dynfu/half.hpp>
#include <dynfu/kinect_fusion_opencl_update_reconstruction_pipeline_block.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <utility>

namespace dynfu {
	
	
	static boost::compute::kernel get_tsdf_kernel (opencl_program_factory & opf) {
		
		auto p=opf("tsdf");
		return boost::compute::kernel(p,"tsdf_kernel");
		
	}	
	
	
	kinect_fusion_opencl_update_reconstruction_pipeline_block::kinect_fusion_opencl_update_reconstruction_pipeline_block (
		boost::compute::command_queue q,
		opencl_program_factory & opf,
		float mu,
		std::size_t tsdf_width,
		std::size_t tsdf_height,
		std::size_t tsdf_depth,
		float tsdf_extent_w,
		float tsdf_extent_h,
		float tsdf_extent_d)
		:	ve_(std::move(q)),
			tsdf_kernel_(get_tsdf_kernel(opf)),
			t_g_k_vec_buf_(ve_.command_queue().get_context(),sizeof(Eigen::Vector3f),CL_MEM_READ_ONLY),
			ik_buf_(ve_.command_queue().get_context(),sizeof(Eigen::Matrix3f),CL_MEM_READ_ONLY),
			k_buf_(ve_.command_queue().get_context(),sizeof(Eigen::Matrix3f),CL_MEM_READ_ONLY),
			proj_view_buf_(ve_.command_queue().get_context(),sizeof(Eigen::Matrix4f),CL_MEM_READ_ONLY),
			weights_(tsdf_width*tsdf_height*tsdf_depth,ve_.command_queue().get_context()),
			mu_(mu),
			tsdf_width_(tsdf_width),
			tsdf_height_(tsdf_height),
			tsdf_depth_(tsdf_depth),
			invk_(0)
	{

		tsdf_kernel_.set_arg(5,proj_view_buf_);
		tsdf_kernel_.set_arg(6,k_buf_);
		tsdf_kernel_.set_arg(7,ik_buf_);
		tsdf_kernel_.set_arg(8,t_g_k_vec_buf_);
		tsdf_kernel_.set_arg(12, tsdf_extent_w);
		tsdf_kernel_.set_arg(13, tsdf_extent_h);
		tsdf_kernel_.set_arg(14, tsdf_extent_d);
		tsdf_kernel_.set_arg(16, weights_);


	}
	
	
	kinect_fusion_opencl_update_reconstruction_pipeline_block::value_type kinect_fusion_opencl_update_reconstruction_pipeline_block::operator () (
		depth_device::value_type::element_type & frame,
		std::size_t frame_width,
		std::size_t frame_height,
		Eigen::Matrix3f k,
		pose_estimation_pipeline_block::value_type::element_type & T_g_k,
		value_type v
	) {
		
		auto && depth_frame = ve_(frame);
		auto q = ve_.command_queue();
		auto t_g_k=T_g_k.get();

		if (t_g_k_ != t_g_k) {
			
			t_g_k_ = std::move(t_g_k);

			// enqueue the proj_view matrix
			Eigen::Matrix4f proj_view = t_g_k_->inverse();
			proj_view.transposeInPlace();

			q.enqueue_write_buffer(proj_view_buf_, 0, sizeof(proj_view), proj_view.data());
			
			// t_g_k is the transformation component of the sensor pose estimation
			Eigen::Vector3f t_g_k_vec = t_g_k_->block<3,1>(0,3);

			q.enqueue_write_buffer(t_g_k_vec_buf_,0,sizeof(t_g_k_vec),t_g_k_vec.data());
		
		}


		if (k_ != k) {

			k_ = k;

			Eigen::Matrix3f k_inv = k.inverse();
			k_inv.transposeInPlace();
			q.enqueue_write_buffer(ik_buf_,0,sizeof(k_inv),k_inv.data());

			k.transposeInPlace();
			q.enqueue_write_buffer(k_buf_,0,sizeof(k),k.data());

		}


		auto && tsdf_ptr = v.buffer;
		using type = opencl_vector_pipeline_value<dynfu::half>;
		if (!tsdf_ptr) tsdf_ptr = std::make_unique<type>(q);
		auto && tsdf_buf = dynamic_cast<type &>(*tsdf_ptr).vector();
		tsdf_buf.resize(tsdf_width_*tsdf_height_*tsdf_depth_, q);
		
		// Set kernel args
		tsdf_kernel_.set_arg(0, depth_frame);
		tsdf_kernel_.set_arg(1, tsdf_buf);
		tsdf_kernel_.set_arg(2, std::uint32_t(tsdf_width_));
		tsdf_kernel_.set_arg(3, std::uint32_t(tsdf_height_));
		tsdf_kernel_.set_arg(4, std::uint32_t(tsdf_depth_));
		// 5,6,7 are bound in constructor
		tsdf_kernel_.set_arg(9, mu_);
		tsdf_kernel_.set_arg(10, std::uint32_t(frame_width));
		tsdf_kernel_.set_arg(11, std::uint32_t(frame_height));
		tsdf_kernel_.set_arg(15, std::uint32_t(invk_++));

		
		// Ready to run the kernel
		std::size_t tsdf_extent[] = {tsdf_width_, tsdf_height_, tsdf_depth_};
		q.enqueue_nd_range_kernel(tsdf_kernel_,3,nullptr,tsdf_extent,nullptr);
		
		
		v.width = tsdf_width_;
		v.height = tsdf_height_;
		v.depth = tsdf_depth_;
		
		return v;
	}
	
}
