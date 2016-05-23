/**
 *	\file
 */


#pragma once


#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/kernel.hpp>
#include <dynfu/measurement_pipeline_block.hpp>
#include <dynfu/opencl_program_factory.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <dynfu/optional.hpp>
#include <cstddef>
#include <Eigen/Dense>


namespace dynfu {
	
	
	/**
	 *	A \ref measurement_pipeline_block implementation as per the
	 *	Kinect Fusion paper.
	 *
	 *	\sa kinect_fusion
	 */
	class kinect_fusion_opencl_measurement_pipeline_block : public measurement_pipeline_block {
		
		
		private:
		
		
			opencl_vector_pipeline_value_extractor<float> ve_;
			boost::compute::kernel bilateral_kernel_;
			boost::compute::kernel v_kernel_;
			boost::compute::kernel n_kernel_;
			boost::compute::vector<float> v_;
			boost::compute::buffer kbuf_;
			optional<Eigen::Matrix3f> k_;
		
		
		public:
		
		
			kinect_fusion_opencl_measurement_pipeline_block () = delete;
			
			
			/**
			 *	Creates a new kinect_fusion_opencl_measurement_pipeline_block.
			 *
			 *	\param [in] q
			 *		A boost::compute::command_queue which the newly created
			 *		object shall use to dispatch OpenCL tasks.
			 *	\param [in] opf
			 *		An \ref opencl_program_factory which the newly created
			 *		object shall use to obtain OpenCL programs.
			 *	\param [in] window_size
			 *		The window size which shall be used by the bilateral
			 *		filter.
			 *	\param [in] sigma_s
			 *		The \f$\sigma_s\f$ value which shall be used by the
			 *		bilateral filter.
			 *	\param [in] sigma_r
			 *		The \f$\sigma_r\f$ value which shall be used by the
			 *		bilateral filter.
			 */
			kinect_fusion_opencl_measurement_pipeline_block (
				boost::compute::command_queue q,
				opencl_program_factory & opf,
				std::size_t window_size,
				float sigma_s,
				float sigma_r
			);
			
			
			/**
			 *	Obtains vertex and normal maps for a certain
			 *	depth frame.
			 *
			 *	\param [in] frame
			 *		A \ref pipeline_value representing the depth
			 *		frame.
			 *	\param [in] width
			 *		The width of \em frame.
			 *	\param [in] height
			 *		The height of \em frame.
			 *	\param [in] k
			 *		The \f$K\f$ matrix for the \ref depth_device
			 *		which generated \em frame.
			 *	\param [in] v
			 *		A std::unique_ptr to a \ref pipeline_value which
			 *		this object may use rather than allocating a new
			 *		one.  Defaults to a default constructed std::unique_ptr
			 *		which will cause this object to allocate a new
			 *		object to return.
			 *
			 *	\return
			 *		A tuple of \ref pipeline_value objects representing a
			 *		vertex map and corresponding normal map.
			 */
			virtual value_type operator () (depth_device::value_type::element_type & frame, std::size_t width, std::size_t height, Eigen::Matrix3f k, value_type v=value_type{});
		
		
	};
	
	
}
