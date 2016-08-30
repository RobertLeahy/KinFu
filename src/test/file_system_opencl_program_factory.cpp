#include <kinfu/file_system_opencl_program_factory.hpp>


#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>
#include <kinfu/filesystem.hpp>
#include <kinfu/path.hpp>
#include <utility>
#include <catch.hpp>


namespace {
	
	
	class fixture {
		
		
		protected:
		
		
			boost::compute::device dev;
			boost::compute::context ctx;
			kinfu::filesystem::path p;
			
			
		public:
		
		
			fixture () : dev(boost::compute::system::default_device()), ctx(dev), p(kinfu::current_executable_parent_path()) {
				
				p/="..";
				p/="data";
				p/="test";
				p/="file_system_opencl_program_factory";
				
			}
		
		
	};
	
	
}


SCENARIO_METHOD(fixture,"kinfu::file_system_opencl_program_factory instances require that the path passed to their constructor be a directory","[kinfu][opencl_program_factory][file_system_opencl_program_factory]") {
	
	GIVEN("A path which does not point to a directory") {
		
		p/="not_a_dir";
		
		THEN("Attempting to construct a kinfu::file_system_opencl_program_factory therefrom throws an exception") {
			
			CHECK_THROWS(kinfu::file_system_opencl_program_factory(std::move(p),std::move(ctx)));
			
		}
		
	}
	
}


SCENARIO_METHOD(fixture,"kinfu::file_system_opencl_program_factory instances throw exceptions when an attempt is made to retrieve a program which does not exist","[kinfu][opencl_program_factory][file_system_opencl_program_factory]") {
	
	GIVEN("A kinfu::file_system_opencl_program_factory") {
		
		kinfu::file_system_opencl_program_factory f(std::move(p),std::move(ctx));
		
		THEN("Attempting to retrieve a program which does not have a corresponding source file throws an exception") {
			
			CHECK_THROWS(f("does_not_exist"));
			
		}
		
	}
	
}


SCENARIO_METHOD(fixture,"kinfu::file_system_opencl_program_factory instances retrieve boost::compute::program objects when provided with a name which matches a *.cl source file","[kinfu][opencl_program_factory][file_system_opencl_program_factory]") {
	
	GIVEN("A kinfu::file_system_opencl_program_factory") {
		
		kinfu::file_system_opencl_program_factory f(std::move(p),std::move(ctx));
		
		THEN("Attempting to retrieve a program which has a corresponding source file succeeds") {
			
			CHECK_NOTHROW(f("bilateral"));
			
		}
		
	}
	
}
