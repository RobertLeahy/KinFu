/**
 *	\file
 */


#pragma once


#include <Eigen/Dense>
#include <utility>


namespace kinfu {


	/**
	 *	Given a coordinate in camera space, and a camera
	 *	calibration matrix, obtains the pixel coordinate and
	 *	depth.
	 *
	 *	\param [in] camera
	 *		A position in camera space.
	 *	\param [in] k
	 *		A camera calibration matrix.
	 *
	 *	\return
	 *		A pair whose first element is the pixel coordinate
	 *		which corresponds to \em camera and whose second element
	 *		is the depth recorded at that pixel when a point at
	 *		\em camera is observed by a camera whose properties
	 *		are encapsulated by \em k.
	 */
	std::pair<Eigen::Vector2i,float> to_pixel (Eigen::Vector3f camera, Eigen::Matrix3f k) noexcept;
	/**
	 *	Given a pixel coordinate, depth, and a camera calibration
	 *	matrix, obtains the corresponding camera coordinate.
	 *
	 *	\param [in] pixel
	 *		A pixel coordinate.
	 *	\param [in] depth
	 *		The depth of \em pixel.
	 *	\param [in] k
	 *		A camera calibration matrix.
	 *
	 *	\return
	 *		The camera coordinate of the point observed by
	 *		the camera when \em depth was recorded at \em pixel.
	 */
	Eigen::Vector3f to_camera (Eigen::Vector2i pixel, float depth, Eigen::Matrix3f k) noexcept;


}
