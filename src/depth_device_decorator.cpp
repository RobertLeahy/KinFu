#include <dynfu/depth_device_decorator.hpp>
#include <utility>


namespace dynfu {
	
	
	depth_device_decorator::depth_device_decorator (depth_device & dev) noexcept : dev_(dev) {	}
	
	
	depth_device_decorator::value_type depth_device_decorator::operator () (value_type v) {
		
		return dev_(std::move(v));
		
	}
	
	
	std::size_t depth_device_decorator::width () const noexcept {
		
		return dev_.width();
		
	}
	
	
	std::size_t depth_device_decorator::height () const noexcept {
		
		return dev_.height();
		
	}
	
	
	Eigen::Matrix3f depth_device_decorator::k () const noexcept {
		
		return dev_.k();
		
	}
	
	
}
