/**
 *	\file
 */


#pragma once


namespace seng499 {
	
	
	/**
	 *	An abstract base class which, when implemented in a
	 *	derived class provides vertex and normal maps based
	 *	on raw depth maps.
	 *
	 *	\sa kinect_fusion
	 */
	class measurement_pipeline_block {
		
		
		public:
		
		
			measurement_pipeline_block () = default;
			measurement_pipeline_block (const measurement_pipeline_block &) = delete;
			measurement_pipeline_block (measurement_pipeline_block &&) = delete;
			measurement_pipeline_block & operator = (const measurement_pipeline_block &) = delete;
			measurement_pipeline_block & operator = (measurement_pipeline_block &&) = delete;
			
			
			/**
			 *	Allows derived classes to be cleaned up through
			 *	pointer or reference to base.
			 */
			virtual ~measurement_pipeline_block () noexcept;
		
		
	};
	
	
}
