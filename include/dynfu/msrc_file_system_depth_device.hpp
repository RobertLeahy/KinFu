/** 
 *	\file
 */


#pragma once


#include <boost/filesystem.hpp>
#include <dynfu/file_system_depth_device.hpp>
#include <Eigen/Dense>
#include <cstddef>


namespace dynfu {
	
	/**
	 * An implementation of file_system_depth_device_filter
	 * that filters results based on the file format of the
	 * <a href="http://research.microsoft.com/en-us/projects/7-scenes/">MSRC 7-Scene datasets</a>. 
	 *
	 * \sa msrc_file_system_depth_device_frame_factory
	 */
	class msrc_file_system_depth_device_filter: public file_system_depth_device_filter {
		
		
		public:
			
			virtual bool operator () (const boost::filesystem::path &) const override;
				
	};
	
	/**
	 * An implementation of file_system_depth_device_frame_factory
	 * that reads PNGs from the file system based on the file format of the
	 * <a href="http://research.microsoft.com/en-us/projects/7-scenes/">MSRC 7-Scene datasets</a>. 
	 *
	 * \sa msrc_file_system_depth_device_filter
	 */
	class msrc_file_system_depth_device_frame_factory: public file_system_depth_device_frame_factory {

		public:
	
			virtual value_type operator () (const boost::filesystem::path &, value_type) override;
	
			virtual std::size_t width () const noexcept override;

			virtual std::size_t height() const noexcept override;

			virtual Eigen::Matrix3f k () const noexcept override;
		
	};
}
