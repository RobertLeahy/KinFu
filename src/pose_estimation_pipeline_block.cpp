#include <dynfu/pose_estimation_pipeline_block.hpp>


namespace dynfu {
	
	
	pose_estimation_pipeline_block::~pose_estimation_pipeline_block () noexcept {	}
	
	tracking_lost_error::tracking_lost_error () : std::runtime_error("The pose estimation pipeline block has lost tracking.") {}
	
}
