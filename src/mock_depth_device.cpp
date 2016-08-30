#include <kinfu/cpu_pipeline_value.hpp>
#include <kinfu/mock_depth_device.hpp>
#include <memory>
#include <stdexcept>
#include <utility>


namespace kinfu {
	
	
	void mock_depth_device::check (const std::vector<float> & f) const {
		
		if (h_==0) {
			
			if (!f.empty()) throw std::logic_error("Frame must be empty if width and height are zero");
			
			return;
			
		}
		
		auto s=f.size();
		if ((s%h_)!=0) throw std::logic_error("Height and width must both evenly divide the frame size");
		if ((s/h_)!=w_) throw std::logic_error("Height times width must give frame size");
		
	}
	
	
	mock_depth_device::mock_depth_device (Eigen::Matrix3f k, std::size_t width, std::size_t height) : h_(height), w_(width), k_(std::move(k)) {
		
		if ((h_==0)!=(w_==0)) throw std::logic_error("Either width and height must both be zero, or neither may be zero");
		
	}
	
	
	void mock_depth_device::add (std::vector<float> f) {
		
		check(f);
		
		fs_.push_back(std::move(f));
		
	}
	
	
	mock_depth_device::value_type mock_depth_device::operator () (value_type v) {
		
		using type=cpu_pipeline_value<buffer_type>;
		if (!v) v=std::make_unique<type>();
		auto && pv=static_cast<type &>(*v.get());
		
		if (fs_.empty()) throw std::logic_error("No more frames");
		
		auto iter=fs_.begin();
		pv.emplace(std::move(*iter));
		fs_.erase(iter);
		
		return v;
		
	}
	
	
	std::size_t mock_depth_device::width () const noexcept {
		
		return w_;
		
	}
	
	
	std::size_t mock_depth_device::height () const noexcept {
		
		return h_;
		
	}
	
	
	Eigen::Matrix3f mock_depth_device::k () const noexcept {
		
		return k_;
		
	}
	
	
}
