/**
 *	\file
 */


#pragma once

#include <dynfu/half.hpp>
#include <dynfu/depth_device.hpp>
#include <dynfu/pipeline_value.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace dynfu {
	
	
	/**
	 *	An abstract base class which when implemented by
	 *	a derived class takes the current sensor pose
	 *	estimation, as well as the raw depth map and
	 *	updates the global truncated signed distance function
	 *	(TSDF).
	 *
	 *	\sa kinect_fusion
	 */
	class update_reconstruction_pipeline_block {
		
		
		public:
		
			/**
			 *	A value in the TSDF.
			 *
			 *	Each voxel in the tsdf has an associated value, forming a scalar
			 *	field in \f${\rm I\!R^3}\f$.
			 */
			using buffer_type=std::vector<half>;
			
		
			/**
			 *	Represents the TSDF.
			 */
			class value_type {
				
				public:

					/**
					 *	The type used to represent the values of the TSDF.
					 */
					using element_type=pipeline_value<buffer_type>;
					
					/**
					 *	A std::unique_ptr to a pipeline_value that stores the buffer of the TSDF.
					 */
					std::unique_ptr<element_type> buffer;
					
					/**
					 * The width of the TSDF.
					 */
					std::size_t width;
					
					/**
					 *	The height of the TSDF.
					 */
					std::size_t height;
					
					/**
					 *	The depth of the TSDF.
					 */
					std::size_t depth;
				
			};
	
		
			update_reconstruction_pipeline_block () = default;
			update_reconstruction_pipeline_block (const update_reconstruction_pipeline_block &) = delete;
			update_reconstruction_pipeline_block (update_reconstruction_pipeline_block &&) = delete;
			update_reconstruction_pipeline_block & operator = (const update_reconstruction_pipeline_block &) = delete;
			update_reconstruction_pipeline_block & operator = (update_reconstruction_pipeline_block &&) = delete;
			
			
			/**
			 *	Allows derived classes to be cleaned up
			 *	through pointer or reference to base.
			 */
			virtual ~update_reconstruction_pipeline_block () noexcept;


			/**
			 *	Updates the global reconstruction of the scene with the
			 *	current raw depth frame data and returns the local TSDF.
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
			 *	\param [in] T_g_k
			 *		A \ref pipeline_value representing the \f$T_{g,k}\f$
			 *		matrix for the current frame, \f$k\f$.
			 *	\param [in] v
			 *		A std::unique_ptr to a \ref pipeline_value which
			 *		this object may use rather than allocating a new
			 *		one.  Defaults to a default constructed std::unique_ptr
			 *		which will cause this object to allocate a new
			 *		object to return.
			 *
			 *	\return
			 *		A \ref pipeline_value representing the TSDF generated from \em frame.
			 */
			virtual value_type operator () (depth_device::value_type::element_type & frame, std::size_t width, std::size_t height, Eigen::Matrix3f k, pose_estimation_pipeline_block::value_type::element_type & T_g_k, value_type v=value_type{}) = 0;
		
	};
	
	
}
