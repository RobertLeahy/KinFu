#include <seng499/fps_depth_device.hpp>
#include <chrono>
#include <thread>


namespace seng499 {
	
	
	static std::chrono::duration<float> fps_to_period (unsigned fps) noexcept {
		
		return std::chrono::duration<float>(1.0/fps);
	
	}
	
	
	fps_depth_device::fps_depth_device (depth_device & dev, unsigned max_fps) noexcept
		:	dev_(dev),
			period_(std::chrono::duration_cast<clock::duration>(fps_to_period(max_fps)))
	{	}
	
	
	std::vector<float> fps_depth_device::operator () (std::vector<float> vec) {
		
		std::this_thread::sleep_until(last_+period_);
		last_=clock::now();
		
		return dev_(std::move(vec));
		
	}
	
	
	std::size_t fps_depth_device::width () const noexcept {
		
		return dev_.width();
		
	}
	
	
	std::size_t fps_depth_device::height () const noexcept {
		
		return dev_.height();
		
	}
	
	
	Eigen::Matrix3f fps_depth_device::k () const noexcept {
		
		return dev_.k();
		
	}
	
	
}
