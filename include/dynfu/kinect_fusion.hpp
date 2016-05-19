/**
 *	\file
 */


#pragma once


#include <dynfu/depth_device.hpp>
#include <dynfu/measurement_pipeline_block.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <dynfu/surface_prediction_pipeline_block.hpp>
#include <dynfu/update_reconstruction_pipeline_block.hpp>


namespace dynfu {
	
	
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
