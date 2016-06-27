/**
 *	\file
 */


#pragma once


#include <Eigen/Dense>
#include <dynfu/pipeline_value.hpp>
#include <dynfu/measurement_pipeline_block.hpp>
#include <dynfu/surface_prediction_pipeline_block.hpp>
#include <memory>

namespace dynfu {
	
	
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
		
		
			/**
			 *	The 4x4 matrix \f$T_{g,k}\f$
			 */
			using sensor_pose_estimation_type=Eigen::Matrix4f;

			/**
			 *	A std::unique_ptr to a \ref pipeline_value
			 *	representing a sensor_pose_estimation_type
			 */
			using value_type=std::unique_ptr<pipeline_value<sensor_pose_estimation_type>>;


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


			/**
			 *	Obtains a new sensor pose estimation \f$T_{g,k}\f$
			 *	from the current vertex and normal maps, registered
			 *	against the previously predicted vertex and normal
			 *	maps.
			 *
			 *	\param [in] live_vn
			 *		A \ref measurement_pipeline_block::value_type representing
			 *		the current vertex and normal maps.
			 *	\param [in] predicted_previous_vn
			 *		A \ref measurement_pipeline_block::value_type representing
			 *		the previously predicted vertex and normal maps.
			 *
			 *	\return
			 *		A \ref value_type object represernting an updated sensor
			 *		pose estimation, \f$T_{g,k}\f$.
			 */
			virtual value_type operator () (
				measurement_pipeline_block::value_type & live_vn,
				surface_prediction_pipeline_block::value_type & predicted_previous_vn,
				Eigen::Matrix3f k,
				Eigen::Matrix4f t_gk_minus_one
			) = 0;
		
		
	};
	
	
}
