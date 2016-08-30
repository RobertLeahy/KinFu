#include <kinfu/fps_depth_device.hpp>
#include <chrono>
#include <thread>


namespace kinfu {
	
	
	static std::chrono::duration<float> fps_to_period (unsigned fps) noexcept {
		
		return std::chrono::duration<float>(1.0/fps);
	
	}
	
	
	fps_depth_device::fps_depth_device (depth_device & dev, unsigned max_fps) noexcept
		:	depth_device_decorator(dev),
			period_(std::chrono::duration_cast<clock::duration>(fps_to_period(max_fps)))
	{	}
	
	
	fps_depth_device::value_type fps_depth_device::operator () (value_type v) {
		
		std::this_thread::sleep_until(last_+period_);
		last_=clock::now();
		
		return dev_(std::move(v));
		
	}
	
	
}
