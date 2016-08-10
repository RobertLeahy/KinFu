/**
 *	\file
 */


#pragma once


#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/kernel.hpp>
#include <dynfu/measurement_pipeline_block.hpp>
#include <dynfu/opencl_program_factory.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <dynfu/optional.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <Eigen/Dense>
#include <cstddef>


namespace dynfu {

	/**
	 *	An implementation of a \ref pose_estimation_pipeline_block
	 *	that uses the OpenCL to calculate a new sensor
	 *	pose estimation.
	 *
	 *	This is the recommended pose_estimation_pipeline_block to use
	 *	in the kinect_fusion pipeline.
	 *
	 *	\sa kinect_fusion
	 */
	class kinect_fusion_opencl_pose_estimation_pipeline_block : public pose_estimation_pipeline_block {


		private:


			boost::compute::command_queue q_;
			boost::compute::kernel corr_;
			boost::compute::kernel parallel_sum_;
			boost::compute::kernel serial_sum_;
			boost::compute::buffer t_z_;
			boost::compute::buffer t_frame_frame_;
			boost::compute::buffer k_;
			boost::compute::buffer mats_;
			boost::compute::buffer mats_output_;
			opencl_vector_pipeline_value_extractor<measurement_pipeline_block::value_type::element_type::type::value_type> e_;
			opencl_vector_pipeline_value_extractor<measurement_pipeline_block::value_type::element_type::type::value_type> p_e_;
			float epsilon_d_;
			float epsilon_theta_;
			std::size_t frame_width_;
			std::size_t frame_height_;
			Eigen::Matrix4f t_gk_initial_;
			std::size_t numit_;
			std::size_t group_size_;
			bool force_px_px_;


		public:


			kinect_fusion_opencl_pose_estimation_pipeline_block () = delete;

			/**
			 *  Creates a new kinect_fusion_opencl_pose_estimation_pipeline_block.
			 *
			 *	\param [in] q
			 *		A boost::compute::command_queue which the newly created
			 *		object shall use to dispatch OpenCL tasks.
			 *
			 *  \param [in] pf
			 *      An \ref opencl_program_factory which the newly created object will
			 *		use to obtain OpenCL programs.
			 *
			 *  \param [in] epsilon_d
			 *      The rejection distance in Equation 17, \f$\epsilon_d\f$
			 *
			 *  \param [in] epsilon_theta
			 *      The normal rejection angle in Equation 17, \f$\epsilon_\theta\f$
			 *
			 *  \param [in] frame_width
			 *      The width of the depth frame
			 *
			 *  \param [in] frame_height
			 *      The height of the depth frame
			 *
			 *	\param [in] t_gk_initial
			 *		The \f$T_{g,k}\f$ to return from the first invocation.
			 *
			 *  \param [in] numit
			 *      The number of iterations to use. The default is 15.
			 *
			 *	\param [in] group_size
			 *		The group size to use in the OpenCL correspondences kernel. Tuning
			 *		this parameter can have a large effect on the runtime of the kernel
			 *		on different hardware. The default is 16. Good results can often be
			 *		obtained with 64 as well.
			 *
			 *	\param [in] force_px_px
			 *		When set to true, forces all correspondences to be pixel to pixel
			 *		instead of generated through projection.
			 */
			kinect_fusion_opencl_pose_estimation_pipeline_block (
				boost::compute::command_queue q,
				opencl_program_factory & pf,
				float epsilon_d,
				float epsilon_theta,
				std::size_t frame_width,
				std::size_t frame_height,
				Eigen::Matrix4f t_gk_initial,
				std::size_t numit=15,
				std::size_t group_size=16,
				bool force_px_px=false
			);


			virtual value_type operator () (
				measurement_pipeline_block::value_type::element_type &,
				measurement_pipeline_block::value_type::element_type *,
				Eigen::Matrix3f,
				value_type
			) override;


	};


}
