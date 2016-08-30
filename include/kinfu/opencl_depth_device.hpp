/**
 *	\file
 */


#pragma once


#include <boost/compute/command_queue.hpp>
#include <kinfu/depth_device_decorator.hpp>


namespace kinfu {
	
	
	/**
	 *	Decorates a \ref depth_device and uploads all
	 *	generated depth frames to the GPU.
	 */
	class opencl_depth_device final : public depth_device_decorator {
		
		
		private:
		
		
			boost::compute::command_queue q_;
			
			
		public:
		
		
			opencl_depth_device () = delete;
			
			
			/**
			 *	Creates a new opencl_depth_device.
			 *
			 *	\param [in] dev
			 *		The \ref depth_device which the newly
			 *		created object shall decorate.
			 *	\param [in] q
			 *		The boost::compute::command_queue which
			 *		the newly created opencl_depth_device shall
			 *		use to dispatch commands to the GPU.
			 */
			opencl_depth_device (depth_device & dev, boost::compute::command_queue q);
			
			
			virtual value_type operator () (value_type v=value_type{}) override;
		
		
	};
	
	
}
