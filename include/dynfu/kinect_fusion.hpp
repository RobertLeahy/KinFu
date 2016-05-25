/**
 *	\file
 */


#pragma once


#include <dynfu/depth_device.hpp>
#include <dynfu/measurement_pipeline_block.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <dynfu/surface_prediction_pipeline_block.hpp>
#include <dynfu/timer.hpp>
#include <dynfu/update_reconstruction_pipeline_block.hpp>
#include <Eigen/Dense>
#include <cstddef>


namespace dynfu {
	
	
	/**
	 *	Represents the KinectFusion processing pipeline.
	 *
	 *	\sa <a href="http://homes.cs.washington.edu/~newcombe/papers/newcombe_etal_ismar2011.pdf">KinectFusion: Real-Time Dense Surface Mapping and Tracking</a>
	 */
	class kinect_fusion {
		
		
		public:
		
		
			/**
			 *	The timer that the pipeline shall use for
			 *	benchmarking.
			 */
			using timer=dynfu::timer;
		
		
		private:
		
		
			dynfu::depth_device * dd_;
			timer::duration ddt_;
			Eigen::Matrix3f k_;
			std::size_t width_;
			std::size_t height_;
			dynfu::measurement_pipeline_block * mpb_;
			timer::duration mpbt_;
			pose_estimation_pipeline_block * pepb_;
			surface_prediction_pipeline_block * sppb_;
			update_reconstruction_pipeline_block * urpb_;
			dynfu::depth_device::value_type frame_;
			dynfu::measurement_pipeline_block::vertex_value_type v_;
			dynfu::measurement_pipeline_block::normal_value_type n_;
			
			
			void get_frame ();
			void get_vertex_and_normal_maps ();
		
		
		public:
		
		
			kinect_fusion (const kinect_fusion &) = delete;
			kinect_fusion (kinect_fusion &&) = delete;
			kinect_fusion & operator = (const kinect_fusion &) = delete;
			kinect_fusion & operator = (kinect_fusion &&) = delete;
			
			
			/**
			 *	Creates a kinect_fusion object with no
			 *	pipeline blocks or depth device.
			 */
			kinect_fusion () noexcept;
			
			
			/**
			 *	Sets the depth device.
			 *
			 *	\param [in] dd
			 *		A reference to the depth device this
			 *		kinect_fusion shall use.  This reference
			 *		must remain valid as long as this object
			 *		is in use or the behaviour is undefined.
			 */
			void depth_device (dynfu::depth_device & dd) noexcept;
			/**
			 *	Sets the measurement pipeline block.
			 *
			 *	\param [in] mpb
			 *		A reference to the measurement pipeline
			 *		block this kinect_fusion shall use.  This
			 *		reference must remain valid as long as this
			 *		object is in use or the behaviour is undefined.
			 */
			void measurement_pipeline_block (dynfu::measurement_pipeline_block & mpb) noexcept;
			
			
			/**
			 *	Retrieves the next frame from the depth device
			 *	and updates the internal state of this object
			 *	accordingly.
			 */
			void operator () ();
			
			
			/**
			 *	Retrieves the amount of time the pipeline spent
			 *	waiting for the depth frame on the last invocation.
			 *
			 *	\return
			 *		The amount of time.
			 */
			timer::duration depth_device_elapsed () const;
			/**
			 *	Retrieves the amount of time the pipeline spent
			 *	waiting for the measurement pipeline block on the
			 *	last invocation.
			 *
			 *	\return
			 *		The amount of time.
			 */
			timer::duration measurement_pipeline_block_elapsed () const;
		
		
	};
	
	
}
