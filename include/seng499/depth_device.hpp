/** 
 *	\file
 */


#pragma once


#include <seng499/pipeline_value.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <vector>


namespace seng499 {
	
	
	/**
	 *	An abstract base class which when implemented
	 *	in a derived class acts as a source of depth
	 *	information.
	 */
	class depth_device {
		
		
		public:
		
		
			/**
			 *	A vector of single precision floating point values
			 *	which contains a raw depth buffer.
			 */
			using buffer_type=std::vector<float>;
			/**
			 *	A std::unique_ptr to a \ref pipeline_value which
			 *	represents a raw depth buffer.
			 */
			using value_type=std::unique_ptr<pipeline_value<buffer_type>>;
			
			
			depth_device () = default;
			depth_device (const depth_device &) = delete;
			depth_device (depth_device &&) = delete;
			depth_device & operator = (const depth_device &) = delete;
			depth_device & operator = (depth_device &&) = delete;
			
			
			/**
			 *	Allows base classes to be cleaned up through
			 *	pointer or reference to base.
			 */
			virtual ~depth_device () noexcept;
			
			
			/**
			 *	Retrieves a frame from the depth sensor.
			 *
			 *	All measurements are in meters.
			 *
			 *	\param [in] v
			 *		A \ref pipeline_value object representing a
			 *		vector of floats which may be used rather
			 *		than allocating a new such object.  Defaults
			 *		to a default constructed std::unique_ptr.
			 *		If this object does not wrap a null pointer
			 *		then the pointee must be a pointee which was
			 *		returned by a previous call to this operator
			 *		on this object or the behaviour is undefined.
			 *
			 *	\return
			 *		A \ref pipeline_value object representing a
			 *		vector of floats.  The vector of floats is
			 *		a row major collection of floating point values
			 *		representing the distance of that pixel from
			 *		the sensor in meters.
			 */
			virtual value_type operator () (value_type v=value_type{}) = 0;
			
			
			/**
			 *	Determines the width of a frame.
			 *
			 *	\return
			 *		The width of a frame in pixels.
			 */
			virtual std::size_t width () const noexcept = 0;
			/**
			 *	Determines the height of a frame.
			 *
			 *	\return
			 *		The height of a frame in pixels.
			 */
			virtual std::size_t height () const noexcept = 0;
			
			
			/**
			 *	Retrieves the camera calibration matrix.
			 *
			 *	The camera calibration matrix transforms points
			 *	on the sensor plane into image pixels, i.e.
			 *	\f$K^{-1}\f$ is the matrix that transforms image
			 *	pixels into their respective voxels in 3-space.
			 *
			 *	\return
			 *		The camera calibration matrix.
			 */
			virtual Eigen::Matrix3f k () const noexcept = 0;
		
		
	};
	
	
}
