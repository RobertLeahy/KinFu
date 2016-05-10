/** 
 *	\file
 */


#pragma once


#include <seng499/file_system_depth_device.hpp>
#include <Eigen/Dense>
#include <regex>


namespace seng499 {
	
	
	class msrc_file_system_depth_device_filter: public file_system_depth_device_filter {
		
		
		public:
			
			virtual bool operator () (const boost::filesystem::path &) const override;
				
	};
	
	class msrc_file_system_depth_device_frame_factory: public file_system_depth_device_frame_factory {
		
		virtual std::vector<float> operator () (const boost::filesystem::path & path, std::vector<float> vec);
		
		virtual std::size_t width () const noexcept;
		
		virtual std::size_t height() const noexcept;
		
		virtual Eigen::Matrix3f k () const noexcept;
		
	};
}
