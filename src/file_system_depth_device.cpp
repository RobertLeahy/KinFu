#include <seng499/file_system_depth_device.hpp>


namespace seng499 {
	
	
	file_system_depth_device::~file_system_depth_device () noexcept {	}
	
	
	file_system_depth_device::file_system_depth_device(const int max_fps) : max_fps_(max_fps), last_invocation_(0) {  } 
	
	
	std::vector<float> file_system_depth_device::operator () (std::vector<float> vec) const {
		
		// if this is not the first invocation, wait until we've at least passed the max fps
		while (last_invocation_ != 0 && (clock() - last_invocation_)/CLOCKS_PER_SEC < (1.0/max_fps_)) { }
		
		// update our last invocation
		last_invocation_ = clock();
		
		// OK, you can have a frame now
		return get_file_system_frame(vec);
	}
	
}
