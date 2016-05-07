/**
 *	\file
 */


#pragma once


namespace seng499 {
	
	
	/**
	 *	An abstract base class which when implemented by
	 *	a derived class provides a sensor pose estimation
	 *	based on the vertex and normal maps of the current
	 *	frame, the global vertex and normal maps, and the
	 *	previous pose estimation, if any.
	 *
	 *	\sa kinect_fusion
	 */
	class pose_estimation_pipeline_block {
		
		
		public:
		
		
			pose_estimation_pipeline_block () = default;
			pose_estimation_pipeline_block (const pose_estimation_pipeline_block &) = delete;
			pose_estimation_pipeline_block (pose_estimation_pipeline_block &&) = delete;
			pose_estimation_pipeline_block & operator = (const pose_estimation_pipeline_block &) = delete;
			pose_estimation_pipeline_block & operator = (pose_estimation_pipeline_block &&) = delete;
			
			
			/**
			 *	Allows derived classes to be cleaned up through
			 *	pointer or reference to base.
			 */
			virtual ~pose_estimation_pipeline_block () noexcept;
		
		
	};
	
	
}
