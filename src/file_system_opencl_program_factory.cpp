#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/opencl_build_error.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>


namespace dynfu {
	
	
	file_system_opencl_program_factory::file_system_opencl_program_factory (filesystem::path path, boost::compute::context ctx) : root_(std::move(path)), ctx_(std::move(ctx)) {
		
		if (filesystem::is_directory(root_)) return;
		
		std::ostringstream ss;
		ss << root_.string() << " is not a directory";
		throw std::logic_error(ss.str());
		
	}
	
	
	boost::compute::program file_system_opencl_program_factory::operator () (const std::string & name) {
		
		auto p=root_;
		p/=name;
		p+=".cl";
	
		auto retr=boost::compute::program::create_with_source_file(p.string(),ctx_);
		opencl_build_error::build(retr);
		
		return retr;
		
	}
	
	
}
