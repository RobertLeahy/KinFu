/** 
 *	\file
 */


#pragma once


#include <Eigen/Dense>
#include <cstddef>
#include <vector>


namespace seng499 {
	
	
	/**
	 *	An abstract base class which when implemented
	 *	in a derived class acts as a source of depth
	 *	information.
	 */
	class depth_device {
		
		
		public:
		
		
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
			 *	\param [in] vec
			 *		An empty vector of floats whose storage may
			 *		be used for the return value.  Should be
			 *		empty.  Defaults to a default constructed
			 *		vector.
			 *
			 *	\return
			 *		A row major collection of floating point values
			 *		representing the distance of that pixel from
			 *		the sensor in meters.
			 */
			virtual std::vector<float> operator () (std::vector<float> vec=std::vector<float>{}) = 0;
			
			
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
