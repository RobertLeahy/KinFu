#include <dynfu/opencl_buffer_pipeline_value.hpp>


#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <catch.hpp>


namespace {
	
	
	class fixture {
		
		
		protected:
		
		
			boost::compute::device dev;
			boost::compute::context ctx;
			boost::compute::command_queue q;
			
			
		public:
		
		
			fixture () : dev(boost::compute::system::default_device()), ctx(dev), q(ctx,dev) {	}
		
		
	};
	
	
}


SCENARIO_METHOD(fixture,"dynfu::opencl_buffer_pipeline_value objects keep the CPU copy up to date with the OpenCL copy as needed","[seng499][pipeline_value][opencl_pipeline_value][opencl_buffer_pipeline_value]") {
	
	GIVEN("An empty dynfu::opencl_buffer_pipeline_value") {
		
		dynfu::opencl_buffer_pipeline_value<int> pv(q);
		
		//	Would test initial value but that doesn't make
		//	any sense
		
		WHEN("Data is transferred to the underlying boost::compute::buffer returned by dynfu::opencl_buffer_pipeline_value::buffer") {
			
			int i=4;
			q.enqueue_write_buffer(pv.buffer(),0,sizeof(i),&i);
			
			THEN("The CPU copy is updated when it is retrieved") {
				
				CHECK(pv.get()==i);
				
			}
			
		}
		
	}
	
}


SCENARIO_METHOD(fixture,"dynfu::opencl_buffer_pipeline_value_extractor objects allow pipeline_value objects to be transparently used on the GPU","[seng499][pipeline_value][opencl_buffer_pipeline_value_extractor]") {
	
	GIVEN("A dynfu::opencl_buffer_pipeline_value_extractor") {
		
		dynfu::opencl_buffer_pipeline_value_extractor<int> e(q);
		
		GIVEN("A dynfu::opencl_vector_pipeline_value associated with the same boost::compute::command_queue") {
			
			dynfu::opencl_buffer_pipeline_value<int> pv(q);
			
			WHEN("The dynfu::opencl_buffer_pipeline_value_extractor is invoked and given the dynfu::opencl_buffer_pipeline_value") {
				
				auto && b=e(pv);
				
				THEN("The boost::compute::buffer wrapped by the dynfu::opencl_buffer_pipeline_value is returned") {
					
					CHECK(&b==&pv.buffer());
					
				}
				
			}
			
		}
		
		GIVEN("A dynfu::opencl_buffer_pipeline_value associated with a different boost::compute::context") {
			
			boost::compute::context n_ctx(dev);
			REQUIRE(n_ctx!=ctx);
			boost::compute::command_queue n_q(n_ctx,dev);
			REQUIRE(n_q!=q);
			dynfu::opencl_buffer_pipeline_value<int> pv(n_q);
			int i=4;
			auto && pvb=pv.buffer();
			n_q.enqueue_write_buffer(pvb,0,sizeof(i),&i);
			
			WHEN("The dynfu::opencl_buffer_pipeline_value_extractor is invoked and given the dynfu::opencl_buffer_pipeline_value") {
				
				auto && b=e(pv);
				
				THEN("A different boost::compute::buffer than that wrapped by the dynfu::opencl_buffer_pipeline_value is returned") {
					
					CHECK(&b!=&pvb);
					
					AND_THEN("The contents of the two buffers are identical") {
						
						REQUIRE(b.size()==sizeof(int));
						REQUIRE(pvb.size()==sizeof(int));
						int pvi;
						n_q.enqueue_read_buffer(pvb,0,sizeof(pvi),&pvi);
						CHECK(pvi==i);
						
					}
					
				}
				
			}
			
		}
		
		GIVEN("A dynfu::cpu_pipeline_value") {
			
			dynfu::cpu_pipeline_value<int> pv;
			pv.emplace(4);
			
			WHEN("The dynfu::opencl_buffer_pipeline_value_extractor is invoked and given the dynfu::cpu_pipeline_value") {
				
				auto && b=e(pv);
				
				THEN("The returned boost::compute::buffer contains the same data as the dynfu::cpu_pipeline_value") {
					
					int i;
					q.enqueue_read_buffer(b,0,sizeof(i),&i);
					CHECK(i==pv.get());
					
				}
				
			}
			
		}
		
	}
	
}
