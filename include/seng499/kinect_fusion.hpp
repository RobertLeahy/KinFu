/**
 *	\file
 */


#pragma once


#include <seng499/depth_device.hpp>
#include <seng499/measurement_pipeline_block.hpp>
#include <seng499/pose_estimation_pipeline_block.hpp>
#include <seng499/surface_prediction_pipeline_block.hpp>
#include <seng499/update_reconstruction_pipeline_block.hpp>


namespace seng499 {
	
	
	/**
	 *	Represents the KinectFusion processing pipeline.
	 *
	 *	\sa <a href="http://homes.cs.washington.edu/~newcombe/papers/newcombe_etal_ismar2011.pdf">KinectFusion: Real-Time Dense Surface Mapping and Tracking</a>
	 */
	class kinect_fusion {
		
		
		private:
		
		
			const depth_device * dd_;
			const measurement_pipeline_block * mpb_;
			const pose_estimation_pipeline_block * pepb_;
			const surface_prediction_pipeline_block * sppb_;
			const update_reconstruction_pipeline_block * urpb_;
		
		
		public:
		
		
			kinect_fusion (const kinect_fusion &) = delete;
			kinect_fusion (kinect_fusion &&) = delete;
			kinect_fusion & operator = (const kinect_fusion &) = delete;
			kinect_fusion & operator = (kinect_fusion &&) = delete;
			
			
			kinect_fusion () noexcept;
		
		
	};
	
	
}
