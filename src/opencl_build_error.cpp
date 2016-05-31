#include <boost/compute.hpp>
#include <dynfu/opencl_build_error.hpp>
#include <sstream>
#include <string>


namespace dynfu {
	
	
	static std::string make_what (const boost::compute::program & p, const boost::compute::opencl_error & err) {
		
		std::ostringstream ss;
		ss << boost::compute::opencl_error::to_string(err.error_code()) << ":\n" << p.build_log();
		
		return ss.str();
		
	}
	
	
	opencl_build_error::opencl_build_error (const boost::compute::program & p, const boost::compute::opencl_error & err)
		:	boost::compute::opencl_error(err),
			p_(p),
			w_(make_what(p,err))
	{   }
	
	
	void opencl_build_error::raise (const boost::compute::program & p, const boost::compute::opencl_error & err) {
		
		switch (err.error_code()) {
			
			case CL_BUILD_PROGRAM_FAILURE:
				break;
			default:
				throw err;
			
		}
		
		throw opencl_build_error(p,err);
		
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
	
	
}
