/**
 *	\file
 */


#pragma once


namespace seng499 {
	
	
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
		
		
	};
	
	
}
