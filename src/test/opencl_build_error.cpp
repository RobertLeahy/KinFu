#include <dynfu/opencl_build_error.hpp>


#include <boost/compute.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/path.hpp>
#include <catch.hpp>


namespace {
	
	
	class fixture {
		
		
		protected:
		
		
			boost::compute::device dev;
			boost::compute::context ctx;
			boost::compute::command_queue q;
			
		public:
		
		
			fixture () : dev(boost::compute::system::default_device()), ctx(dev) {	}
			
		
	};
	
	
}


SCENARIO_METHOD(fixture,"dynfu::opencl_build_error::raise throws an appropriate exception","[dynfu][opencl_build_error]") {
	
	GIVEN("A boost::compute::program") {
		
		dynfu::filesystem::path path(dynfu::current_executable_parent_path());
		path/="..";
		path/="data";
		path/="test";
		path/="file_system_opencl_program_factory";
		path/="bilateral.cl";
		auto p=boost::compute::program::create_with_source_file(path.string(),ctx);
		
		GIVEN("A boost::compute::opencl_error which does not represent CL_BUILD_PROGRAM_FAILURE") {
			
			boost::compute::opencl_error err(CL_INVALID_PROGRAM);
			
			THEN("Invoking dynfu::opencl_build_error::raise throws a boost::compute::opencl_error which represents the same error code") {
				
				bool threw=false;
				try {
					
					dynfu::opencl_build_error::raise(p,err);
					
				} catch (const dynfu::opencl_build_error &) {
					
					//	Fail
					
				} catch (const boost::compute::opencl_error & ex) {
					
					threw=true;
					CHECK(ex.error_code()==err.error_code());
					
				}
				
				CHECK(threw);
				
			}
			
		}
		
		GIVEN("A boost::compute::opencl_error which represents CL_BUILD_PROGRAM_FAILURE") {
			
			boost::compute::opencl_error err(CL_BUILD_PROGRAM_FAILURE);
			
			THEN("Invoking dynfu::opencl_build_error::raise throws a dynfu::opencl_build_error which represents CL_BUILD_PROGRAM_FAILURE and contains the appropriate boost::compute::program") {
				
				bool threw=false;
				try {
					
					dynfu::opencl_build_error::raise(p,err);
					
				} catch (const dynfu::opencl_build_error & ex) {
					
					threw=true;
					CHECK(ex.error_code()==CL_BUILD_PROGRAM_FAILURE);
					CHECK(ex.program()==p);
					
				}
				
				CHECK(threw);
				
			}
			
		}
		
	}
	
}


SCENARIO_METHOD(fixture,"dynfu::opencl_file_build_error::raise throws an appropriate exception","[dynfu][opencl_build_error][opencl_file_build_error]") {
	
	GIVEN("A boost::compute::program created with source from a file") {
		
		dynfu::filesystem::path path(dynfu::current_executable_parent_path());
		path/="..";
		path/="data";
		path/="test";
		path/="file_system_opencl_program_factory";
		path/="bilateral.cl";
		auto p=boost::compute::program::create_with_source_file(path.string(),ctx);
		
		GIVEN("A boost::compute::opencl_error which does not represent CL_BUILD_PROGRAM_FAILURE") {
			
			boost::compute::opencl_error err(CL_INVALID_PROGRAM);
			
			THEN("Invoking dynfu::opencl_file_build_error::raise throws a boost::compute::opencl_error which represents the same error code") {
				
				bool threw=false;
				try {
					
					dynfu::opencl_file_build_error::raise(p,err,path);
					
				} catch (const dynfu::opencl_build_error &) {
					
					//	Fail
					
				} catch (const boost::compute::opencl_error & ex) {
					
					threw=true;
					CHECK(ex.error_code()==err.error_code());
					
				}
				
				CHECK(threw);
				
			}
			
		}
		
		GIVEN("A boost::compute::opencl_error which represents CL_BUILD_PROGRAM_FAILURE") {
			
			boost::compute::opencl_error err(CL_BUILD_PROGRAM_FAILURE);
			
			THEN("Invoking dynfu::opencl_file_build_error::raise throws a dynfu::opencl_file_build_error which represents CL_BUILD_PROGRAM_FAILURE and contains the appropriate boost::compute::program and dynfu::filesystem::path") {
				
				bool threw=false;
				try {
					
					dynfu::opencl_file_build_error::raise(p,err,path);
					
				} catch (const dynfu::opencl_file_build_error & ex) {
					
					threw=true;
					CHECK(ex.error_code()==CL_BUILD_PROGRAM_FAILURE);
					CHECK(ex.program()==p);
					CHECK(ex.path()==path);
					
				}
				
				CHECK(threw);
				
			}
			
		}
		
	}
	
}
