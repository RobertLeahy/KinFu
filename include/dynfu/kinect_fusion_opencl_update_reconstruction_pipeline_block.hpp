/**
 *	\file
 */


#pragma once


#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/kernel.hpp>
#include <dynfu/depth_device.hpp>
#include <dynfu/opencl_program_factory.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <dynfu/optional.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <dynfu/update_reconstruction_pipeline_block.hpp>
#include <Eigen/Dense>
#include <cstddef>




namespace dynfu {
	
	
	/**
	 *	An \ref update_reconstruction_pipeline_block implemented
	 *	as per the Kinect Fusion paper.
	 *
	 *	\sa kinect_fusion
	 */
	class kinect_fusion_opencl_update_reconstruction_pipeline_block : public update_reconstruction_pipeline_block {
		
		
		private:
			opencl_vector_pipeline_value_extractor<float> ve_;
			boost::compute::kernel tsdf_kernel_;
			boost::compute::buffer t_g_k_vec_buf_;
			boost::compute::buffer ik_buf_;
			boost::compute::buffer k_buf_;
			boost::compute::buffer proj_view_buf_;
			optional<Eigen::Matrix4f> t_g_k_;
			optional<Eigen::Matrix3f> k_;
			float mu_;
			
			std::size_t tsdf_width_;
			std::size_t tsdf_height_;
			std::size_t tsdf_depth_;
			
		
		public:
			
			kinect_fusion_opencl_update_reconstruction_pipeline_block () = delete;
			
			/**
			 *	Creates a new kinect_fusion_opencl_update_reconstruction_pipeline_block.
			 *
			 *	\param [in] q
			 *		A boost::compute::command_queue which the newly created
			 *		object shall use to dispatch OpenCL tasks.
			 *	\param [in] opf
			 *		An \ref opencl_program_factory which the newly created
			 *		object shall use to obtain OpenCL programs.
			 *	\param [in] mu
			 *		\f$\mu\f$: the truncation distance of the truncated signed distance function (TSDF).
			 *		Points greater than a distance of \f$\mu\f$ from the nearest surface interface are
			 *		truncated to the maximum distance of \f$\mu\f$. Non-visible points farther than \f$\mu\f$
			 *		from the surface are not measured.
			 *	\param [in] tsdf_width
			 *		The number of elements of the TSDF in the width direction.
			 *	\param [in] tsdf_height
			 *		The number of elements of the TSDF in the height direction.
			 *	\param [in] tsdf_depth
			 *		The number of elements of the TSDF in the depth direction.
			 *	\param [in] tsdf_extent_w
			 *		The extent of the TSDF (in meters) in the width direction.
			 *	\param [in] tsdf_extent_h
			 *		The extent of the TSDF (in meters) in the height direction.
			 *	\param [in] tsdf_extent_d
			 *		The extent of the TSDF (in meters) in the depth direction.
			 */
			kinect_fusion_opencl_update_reconstruction_pipeline_block (				
				boost::compute::command_queue q,
				opencl_program_factory & opf,
				float mu,
				std::size_t tsdf_width=512,
				std::size_t tsdf_height=512,
				std::size_t tsdf_depth=512,
				float tsdf_extent_w=3.0f,
				float tsdf_extent_h=3.0f,
				float tsdf_extent_d=3.0f
			);
			
			virtual value_type operator () (depth_device::value_type::element_type & frame, std::size_t width, std::size_t height, Eigen::Matrix3f k, pose_estimation_pipeline_block::value_type::element_type & T_g_k, value_type v=value_type{});
			 
	};
	
	
}
