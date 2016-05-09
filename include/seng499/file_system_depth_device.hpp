/** 
 *	\file
 */


#pragma once

#include <seng499/depth_device.hpp>

#include <Eigen/Dense>
#include <cstddef>
#include <vector>
#include <ctime>


namespace seng499 {
	
	
	/**
	 *	An abstract base class which when implemented
	 *  in a derived class acts as a source of pre-existing
     *  depth information located on a file system.
	 * 
	 *  Includes provisions to allow for fps throttling. 
	 */
	class file_system_depth_device: public depth_device {
		
		
		
		
		public:
		
			
			/**
			 *	Allows base classes to be cleaned up through
			 *	pointer or reference to base.
			 */
			virtual ~file_system_depth_device () noexcept;
			
			
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
			 virtual std::vector<float> operator () (std::vector<float> vec) const override final;
			 
            
		private:
			
			int max_fps_;
			mutable clock_t last_invocation_;
            
		protected:
		
		
			/**
			 * Creates a new file_system_depth_device.
			 *
			 * \param [in] max_fps
			 * 		The max number of frames per second that
			 *		this file_system_depth_device provides
			 *
			 */
			file_system_depth_device (const int max_fps); 
		
		
			/**
			 *	A template method allowing for the fps to be
			 *  debounced on a file_system_depth_device
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
			virtual std::vector<float> get_file_system_frame(std::vector<float> vec) const = 0;
	};
	
	
}
