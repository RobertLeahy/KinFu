#include <seng499/opencl_vector_pipeline_value.hpp>


#include <boost/compute.hpp>
#include <algorithm>
#include <iterator>
#include <vector>
#include <catch.hpp>


namespace {
	
	
	class fixture {
		
		
		protected:
		
		
			boost::compute::device dev;
			boost::compute::context ctx;
			boost::compute::command_queue q;
			seng499::opencl_vector_pipeline_value<int> pv;
			
			
		public:
		
		
			fixture () : dev(boost::compute::system::default_device()), ctx(dev), q(ctx,dev), pv(q) {	}
		
		
	};
	
	
}


SCENARIO_METHOD(fixture,"seng499::opencl_vector_pipeline_value objects keep the CPU copy up to date with the GPU copy as needed","[seng499][pipeline_value][opencl_pipeline_value][opencl_vector_pipeline_value]") {
	
	GIVEN("An empty seng499::opencl_vector_pipeline_value") {
		
		THEN("The CPU copy is empty") {
			
			CHECK(pv.get().empty());
			
		}
		
		WHEN("Data is transferred to the underlying boost::compute::vector retrieved through the seng499::opencl_vector_pipeline_value::vector accessor") {
			
			auto && gpu=pv.vector();
			std::vector<int> cpu{1,2,3,4};
			boost::compute::copy(cpu.begin(),cpu.end(),std::back_inserter(gpu),q);
			
			THEN("The CPU copy is updated when it is retrieved again") {
				
				auto && v=pv.get();
				CHECK(std::equal(v.begin(),v.end(),cpu.begin(),cpu.end()));
				
			}
			
		}
		
	}
	
}
