/**
 *	\file
 */


#pragma once


#include <Eigen/Dense>
#include <dynfu/depth_device.hpp>
#include <dynfu/pipeline_value.hpp>
#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>


namespace dynfu {
	
	
	/**
	 *	An abstract base class which, when implemented in a
	 *	derived class provides vertex and normal maps based
	 *	on raw depth maps.
	 *
	 *	\sa kinect_fusion
	 */
	class measurement_pipeline_block {
		
		
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
			
			
			/**
			 *	Obtains vertex and normal maps for a certain
			 *	depth frame.
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
			 *	\param [in] v
			 *		A std::unique_ptr to a \ref pipeline_value which
			 *		this object may use rather than allocating a new
			 *		one.  Defaults to a default constructed std::unique_ptr
			 *		which will cause this object to allocate a new
			 *		object to return.
			 *
			 *	\return
			 *		A tuple of \ref pipeline_value objects representing a
			 *		vertex map and corresponding normal map.
			 */
			virtual value_type operator () (depth_device::value_type::element_type & frame, std::size_t width, std::size_t height, Eigen::Matrix3f k, value_type v=value_type{}) = 0;
		
		
	};
	
	
}
