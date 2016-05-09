/** 
 *	\file
 */


#pragma once


#include <seng499/depth_device.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <cstddef>
#include <vector>


namespace seng499 {
	
	
	/**
	 *	An abstract base class which when implemented
	 *  in a derived class acts as a source of pre-existing
     *  depth information located on a file system.
	 * 
	 *  Includes provisions to allow for fps throttling.
	 */
	class file_system_depth_device : public depth_device {
		
		
		public:
			
			
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
			virtual std::vector<float> operator () (std::vector<float> vec=std::vector<float>{}) const override final;
			
			
			/**
			 * Creates a new file_system_depth_device.
			 *
			 * \param [in] max_fps
			 * 		The max number of frames per second that
			 *		this file_system_depth_device provides
			 *
			 */
			explicit file_system_depth_device (unsigned max_fps) noexcept;
			
            
		private:
		
		
			using clock=std::chrono::high_resolution_clock;
			
			
			clock::duration period_;
			mutable clock::time_point last_invocation_;
			
            
		protected:
		
		
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
