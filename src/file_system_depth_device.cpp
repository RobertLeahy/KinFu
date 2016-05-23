#include <dynfu/file_system_depth_device.hpp>
#include <dynfu/filesystem.hpp>
#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <utility>


namespace dynfu {
	
	
	file_system_depth_device_filter::~file_system_depth_device_filter () noexcept {	}
	
	
	bool null_file_system_depth_device_filter::operator () (const filesystem::path &) const {
		
		return true;
		
	}
	
	
	file_system_depth_device_comparer::~file_system_depth_device_comparer () noexcept {	}
	
	
	bool null_file_system_depth_device_comparer::operator () (const filesystem::path & a, const filesystem::path & b) const {
		
		std::less<> c;
		return c(a,b);
		
	}
	
	
	file_system_depth_device_frame_factory::~file_system_depth_device_frame_factory () noexcept {	}
	
	
	file_system_depth_device::end::end () : std::runtime_error("No more files") {	}
	
	
	file_system_depth_device::file_system_depth_device (filesystem::path path, file_system_depth_device_frame_factory & factory, const file_system_depth_device_filter * filter, const file_system_depth_device_comparer * comparer) : factory_(factory) {
		
		if (!filesystem::is_directory(path)) {
			
			std::ostringstream ss;
			ss << "file_system_depth_device: " << path << " is not a directory";
			throw std::invalid_argument(ss.str());
			
		}
		
		null_file_system_depth_device_comparer nc;
		const file_system_depth_device_comparer & c=comparer ? *comparer : nc;
		null_file_system_depth_device_filter nf;
		const file_system_depth_device_filter & f=filter ? *filter : nf;
		
		std::for_each(filesystem::directory_iterator(path),filesystem::directory_iterator{},[&] (const auto & entry) {
			
			if (!filesystem::is_regular_file(entry.status())) return;
			auto && path=entry.path();
			if (!f(path)) return;
			files_.push_back(path);
			
		});
		
		//	Sorts in reverse so we can pop back to get the "front"
		std::sort(files_.begin(),files_.end(),[&] (const auto & a, const auto & b) {	return !c(a,b);	});
		
	}
	
	
	file_system_depth_device::value_type file_system_depth_device::operator () (value_type v) {
		
		if (files_.empty()) throw end{};
		
		auto path=std::move(files_.back());
		files_.pop_back();
		
		return factory_(path,std::move(v));
		
	}
	
	
	std::size_t file_system_depth_device::width () const noexcept {
		
		return factory_.width();
		
	}
	
	
	std::size_t file_system_depth_device::height () const noexcept {
		
		return factory_.height();
		
	}
	
	
	Eigen::Matrix3f file_system_depth_device::k () const noexcept {
		
		return factory_.k();
		
	}
	
	
	file_system_depth_device::operator bool () const noexcept {
		
		return !files_.empty();
		
	}
	
	
}
