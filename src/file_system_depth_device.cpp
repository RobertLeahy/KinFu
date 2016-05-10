#include <seng499/file_system_depth_device.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>


namespace seng499 {
	
	
	static std::chrono::duration<float> fps_to_period (unsigned fps) noexcept {
		
		return std::chrono::duration<float>(1.0/fps);
	
	}
	
	
	file_system_depth_device::file_system_depth_device(unsigned max_fps, boost::filesystem::path path, 
		optional<boost::regex> regex, bool sort_desc) noexcept : 
		period_(std::chrono::duration_cast<clock::duration>(fps_to_period(max_fps))),
		path_(std::move(path)), regex_(std::move(regex)), sort_desc_(sort_desc) {
		
		if (!boost::filesystem::is_directory(path_)) {
			throw new std::invalid_argument("file_system_depth_device: provided path must be an existing directory");
		}
		
		/* 
		 * Loop over files in the directory specified by path,
		 * filter by the supplied regex if supplied. Then sort.
		 * Finally, if sort_desc_ is specified, reverse the vector.
		 *
		 */
		std::vector<boost::filesystem::path> collected_files;
		
		boost::filesystem::directory_iterator end_itr;
		
		for (boost::filesystem::directory_iterator dir_itr( path_ ); dir_itr != end_itr; ++dir_itr) {
			
			if (boost::filesystem::is_regular_file( dir_itr->status() )) {
				
				if (!regex_ || boost::regex_match(dir_itr->path().filename().string(), regex_.value())) {
					
					collected_files.push_back(boost::filesystem::canonical( dir_itr->path() ));
					
				}
				
			}
			
		}
	
		std::sort(collected_files.begin(), collected_files.end());
		
		if (sort_desc_) {
			std::reverse(collected_files.begin(), collected_files.end());
		}
		
		sorted_files_ = std::move(collected_files);
		
		current_file_ = sorted_files_.begin();
		
	}
	
	
	std::vector<float> file_system_depth_device::operator () (std::vector<float> vec) const {
		
		std::this_thread::sleep_until(last_invocation_+period_);
		last_invocation_=clock::now();
		
		// OK, you can have a frame now
		return get_file_system_frame(std::move(vec));
	}
	
	
}
