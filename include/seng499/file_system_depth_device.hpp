/** 
 *	\file
 */


#pragma once


#include <seng499/depth_device.hpp>
#include <seng499/optional.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <cstddef>
#include <vector>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>


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
			 *	\param [in] path
			 *		A path to the directory where a collection of
			 *		depth frame files exists.
			 *	\param [in] regex 
			 *		An optionally specified string to be converted
			 *		to a boost::regex, or an empty string (default)
			 *		to indicate that no filtering is needed. 
			 *	\param [in] sort_desc
			 *		An optionally specified boolean value indicating 
			 *		that the files in directory path, filtered by regex
			 *		should be sorted in descending order. Default is false.
			 */
			explicit file_system_depth_device (unsigned max_fps, const boost::filesystem::path path, 
					const optional<boost::regex> regex = {}, const bool sort_desc=false) noexcept;
			
            
		private:
		
		
			using clock=std::chrono::high_resolution_clock;
			
			
			clock::duration period_;
			mutable clock::time_point last_invocation_;
			
			boost::filesystem::path path_;
			optional<boost::regex> regex_;
			bool sort_desc_;
			

            
		protected:
		
			std::vector<boost::filesystem::path> sorted_files_;
			
			/**
			 *	Implementations can increment and access current_file_
			 *	to get the next file to process.
			 */
			std::vector<boost::filesystem::path>::iterator current_file_;
		
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
