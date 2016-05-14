#include <boost/filesystem.hpp>
#include <seng499/file_system_opencl_program_factory.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>


namespace seng499 {
	
	
	file_system_opencl_program_factory::file_system_opencl_program_factory (boost::filesystem::path path, boost::compute::context ctx) : root_(std::move(path)), ctx_(std::move(ctx)) {
		
		if (boost::filesystem::is_directory(root_)) return;
		
		std::ostringstream ss;
		ss << root_.string() << " is not a directory";
		throw std::logic_error(ss.str());
		
	}
	
	
	boost::compute::program file_system_opencl_program_factory::operator () (const std::string & name) {
		
		auto p=root_;
		p/=name;
		p+=".cl";
		
		auto retr=boost::compute::program::create_with_source_file(p.string(),ctx_);
		retr.build();
		
		return retr;
		
	}
	
	
}
