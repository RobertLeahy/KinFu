/**
 *	\file
 */


#pragma once


#include <Eigen/Dense>
#include <seng499/depth_device.hpp>
#include <chrono>
#include <stdexcept>


namespace seng499 {
	
	
	/**
	 *	Decorates a \ref depth_device by limiting the rate
	 *	at which frames may be acquired such that a certain
	 *	maximum number of frames per second are allowed.
	 *
	 *	This class is not thread safe.
	 */
	class fps_depth_device final : public depth_device {
		
		
		private:
		
		
			using clock=std::chrono::high_resolution_clock; 
			
			
			depth_device & dev_;
			clock::duration period_;
			clock::time_point last_;
			
			
		public:
		
		
			/**
			 *	Creates a new fps_depth_device.
			 *
			 *	\param [in] dev
			 *		The \ref depth_device to decorate.
			 *	\param [in] max_fps
			 *		The maximum number of invocations of
			 *		\ref depth_device::operator() which will
			 *		be allowed per second.
			 */
			fps_depth_device (depth_device & dev, unsigned max_fps) noexcept;
			
			
			virtual value_type operator () (value_type v=value_type{}) override;
			virtual std::size_t width () const noexcept override;
			virtual std::size_t height () const noexcept override;
			virtual Eigen::Matrix3f k () const noexcept override;
		
		
	};
	
	
}
