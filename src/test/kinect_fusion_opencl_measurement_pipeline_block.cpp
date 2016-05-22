#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>


#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <tuple>
#include <vector>
#include <catch.hpp>


namespace {
	

	class fixture {
		
		
		protected:
		
			boost::compute::device dev;
			boost::compute::context ctx;
			boost::compute::command_queue q;
			std::size_t width;
			std::size_t height;
			Eigen::Matrix3f k;
			dynfu::file_system_opencl_program_factory fsopf;
			
		public:
		
			fixture () : dev(boost::compute::system::default_device()), ctx(dev), q(ctx,dev), width(4), height(4), fsopf("cl",ctx) {
					
				k << 585.0f, 0.0f, 320.0f,
					 0.0f, -585.0f, 240.0f,
					 0.0f, 0.0f, 1.0f;
			
			}
		
	};
	

}


SCENARIO_METHOD(fixture, "A dynfu::kinect_fusion_opencl_measurement_pipeline_block implements the measurement phase of the kinect fusion pipeline on the GPU using OpenCL","[seng499][measurement_pipeline_block][kinect_fusion_opencl_measurement_pipeline_block]") {
	
	GIVEN("A dynfu::kinect_fusion_opencl_measurement_pipeline_block") {		
		
		dynfu::kinect_fusion_opencl_measurement_pipeline_block kfompb(q, fsopf);
		dynfu::cpu_pipeline_value<std::vector<float>> pv;
		std::vector<float> v{0.0f, 5.0f, 10.0f, 15.0f, 13.0f, 40.0f, 12.0f, 10.0f, 8.0f,19.0f,202.0f,102.0f,84.0f,293.0f,292.0f,293.0f};

		pv.emplace(std::move(v));
		
		WHEN("It is invoked") {
			
			kfompb(pv, width, height, k);
			
			THEN("stuff happens!") {
				
				CHECK(1==1);
				
			}
			
		}
		
	}
	
}
