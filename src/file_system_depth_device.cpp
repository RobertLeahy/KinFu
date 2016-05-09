#include <seng499/file_system_depth_device.hpp>
#include <chrono>
#include <thread>
#include <utility>


namespace seng499 {
	
	
	static std::chrono::duration<float> fps_to_period (unsigned fps) noexcept {
		
		return std::chrono::duration<float>(1.0/fps);
	
	}
	
	
	file_system_depth_device::file_system_depth_device(unsigned max_fps) noexcept : period_(std::chrono::duration_cast<clock::duration>(fps_to_period(max_fps))) {	}
	
	
	std::vector<float> file_system_depth_device::operator () (std::vector<float> vec) const {
		
		std::this_thread::sleep_until(last_invocation_+period_);
		last_invocation_=clock::now();
		
		// OK, you can have a frame now
		return get_file_system_frame(std::move(vec));
	}
	
	
}
