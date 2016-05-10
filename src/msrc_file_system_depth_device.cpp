#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <seng499/msrc_file_system_depth_device.hpp>
#include <regex>
#include <iostream>


namespace seng499 {

	static const std::regex & get_regex() {

		static const std::regex msrc_depth_file_pattern(".*depth.*");
		return msrc_depth_file_pattern;

	}
	
	
	bool msrc_file_system_depth_device_filter::operator () (const boost::filesystem::path & path) const {
		
		return std::regex_match(path.filename().string(), get_regex());
		
	}
	
	
	std::vector<float> msrc_file_system_depth_device_frame_factory::operator () (const boost::filesystem::path & path, std::vector<float> vec) {
	
		// Load a grayscale depth image. Anydepth is necessary as these are 16 bit ints 
		cv::Mat i_img = cv::imread(path.native(), CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);

		// convert to floats	
		cv::Mat f_img;
		i_img.convertTo(f_img, CV_32FC1); 

		// OpenCV stores matrices in row order -> copy f_img to vec
		const float* f_img_ptr = f_img.ptr<float>();
		std::size_t elements = f_img.rows * f_img.cols;
		vec.assign(f_img_ptr, f_img_ptr + elements);

		return vec;
		
	}
	
	
	std::size_t msrc_file_system_depth_device_frame_factory::width () const noexcept {
		
		return 640;
		
	}
	
	std::size_t msrc_file_system_depth_device_frame_factory::height () const noexcept {
		
		return 480;
		
	}
	
	Eigen::Matrix3f msrc_file_system_depth_device_frame_factory::k () const noexcept {
		
		Eigen::Matrix3f k;
		k << 585.0f, 0.0f, 320.0f,
		     0.0f, -585.0f, 240.0f,
		     0.0f, 0.0f, 1.0f;
		
		return k;
	}
	
}
