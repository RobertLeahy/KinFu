/**
 *	\file
 */


#pragma once

#include <dynfu/pipeline_value.hpp>

#include <Eigen/Dense>

#include <memory>
#include <tuple>
#include <vector>

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

			/**
			 *	A collection of vertices.
			 *
			 *	Each vertex in a vertex map has a corresponding
			 *	normal in the associated normal map.
			 */
			using vertex_map_type=std::vector<Eigen::Vector3f>;
			/**
			 *	A std::unique_ptr to a \ref pipeline_value
			 *	representing a vertex map.
			 */
			using vertex_value_type=std::unique_ptr<pipeline_value<vertex_map_type>>;
			/**
			 *	A collection of normals.
			 *
			 *	Each normal in a normal map has a corresponding
			 *	vertex in the associated vertex map.
			 */
			using normal_map_type=std::vector<Eigen::Vector3f>;
			/**
			 *	A std::unique_ptr to a \ref pipeline_value
			 *	representing a normal map.
			 */
			using normal_value_type=std::unique_ptr<pipeline_value<normal_map_type>>;
			/**
			 *	The type this pipeline block generates.
			 *
			 *	A tuple whose first element is a vertex map and whose
			 *	second element is the associated normal map.
			 */
			using value_type=std::tuple<vertex_value_type,normal_value_type>;


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
