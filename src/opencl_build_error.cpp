#include <boost/compute.hpp>
#include <kinfu/filesystem.hpp>
#include <kinfu/opencl_build_error.hpp>
#include <sstream>
#include <string>
#include <utility>


namespace kinfu {
	
	
	static std::string make_what (const boost::compute::program & p, const boost::compute::opencl_error & err, const filesystem::path * path=nullptr) {
		
		std::ostringstream ss;
		ss << err.error_string();
		if (path) ss << " in " << path->string();
		auto l=p.build_log();
		//	Sometimes the build log is empty even if the
		//	error is CL_BUILD_PROGRAM_FAILURE
		//
		//	This has been observed specifically on Mac OSX
		//	El Capitan with a Radeon 4xxx card, the build
		//	fails (with CL_LOG_ERRORS=stdout it simply says
		//	that the program couldn't be built for the GPU)
		//	and the build log is empty
		if (!l.empty()) ss << ":\n" << l;
		return ss.str();
		
	}
	
	
	static std::string make_what (const boost::compute::program & p, const boost::compute::opencl_error & err, const filesystem::path & path) {
		
		return make_what(p,err,&path);
		
	}
	
	
	opencl_build_error::opencl_build_error (const boost::compute::program & p, const boost::compute::opencl_error & err)
		:	opencl_build_error(p,err,make_what(p,err))
	{	}
	
	
	opencl_build_error::opencl_build_error (const boost::compute::program & p, const boost::compute::opencl_error & err, std::string w)
		:	boost::compute::opencl_error(err),
			p_(p),
			w_(std::move(w))
	{	}
	
	
	static bool is_build_error (const boost::compute::opencl_error & err) noexcept {
		
		return err.error_code()==CL_BUILD_PROGRAM_FAILURE;
		
	}
	
	
	void opencl_build_error::raise (const boost::compute::program & p, const boost::compute::opencl_error & err) {
		
		if (is_build_error(err)) throw opencl_build_error(p,err);
		throw err;
		
	}
	
	
	void opencl_build_error::build (boost::compute::program & p) {
		
		try {
			
			p.build();
			
		} catch (const boost::compute::opencl_error & err) {
			
			raise(p,err);
			
		}
		
	}
	
	
	boost::compute::program opencl_build_error::program () const {
		
		return p_;
		
	}
	
	
	const char * opencl_build_error::what () const noexcept {
		
		return w_.c_str();
		
	}
	
	
	opencl_file_build_error::opencl_file_build_error (const boost::compute::program & p, const boost::compute::opencl_error & err, const filesystem::path & path)
		:	opencl_build_error(p,err,make_what(p,err,path)),
			p_(path)
	{	}
	
	
	void opencl_file_build_error::raise (const boost::compute::program & p, const boost::compute::opencl_error & err, const filesystem::path & path) {
		
		if (is_build_error(err)) throw opencl_file_build_error(p,err,path);
		throw err;
		
	}
	
	
	boost::compute::program opencl_file_build_error::build (const filesystem::path & path, const boost::compute::context & ctx) {
		
		auto retr=boost::compute::program::create_with_source_file(path.string(),ctx);
		
		try {
			
			retr.build();
			
		} catch (const boost::compute::opencl_error & err) {
			
			raise(retr,err,path);
			
		}
		
		return retr;
		
	}
	
	
	const filesystem::path & opencl_file_build_error::path () const noexcept {
		
		return p_;
		
	}
	
	
}
