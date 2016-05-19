/**
 *	\file
 */


#pragma once


namespace dynfu {
	
	
	/**
	 *	An abstract base class which when implemented in
	 *	a derived class takes as input the truncated
	 *	signed distance function (TSDF) and previous camera
	 *	pose estimation and outputs the new global vertex
	 *	and normal maps.
	 *
	 *	\sa kinect_fusion
	 */
	class surface_prediction_pipeline_block {
		
		
		public:
		
		
			surface_prediction_pipeline_block () = default;
			surface_prediction_pipeline_block (const surface_prediction_pipeline_block &) = delete;
			surface_prediction_pipeline_block (surface_prediction_pipeline_block &&) = delete;
			surface_prediction_pipeline_block & operator = (const surface_prediction_pipeline_block &) = delete;
			surface_prediction_pipeline_block & operator = (surface_prediction_pipeline_block &&) = delete;
			
			
			/**
			 *	Allows derived classes to be cleaned up
			 *	through pointer or reference to base.
			 */
			virtual ~surface_prediction_pipeline_block () noexcept;
		
		
	};
	
	
}
