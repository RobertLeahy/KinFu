#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>


#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <Eigen/Dense>
#include <algorithm>
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
		
		dynfu::kinect_fusion_opencl_measurement_pipeline_block kfompb(q, fsopf, 16, 2.0f, 1.0f);
		dynfu::cpu_pipeline_value<std::vector<float>> pv;
		std::vector<float> v{0.0f, 5.0f, 10.0f, 15.0f, 13.0f, 40.0f, 12.0f, 10.0f, 8.0f,19.0f,202.0f,102.0f,84.0f,293.0f,292.0f,293.0f};


		std::vector<Eigen::Vector3f> vgt {
			
			{-2.95821e-011, 2.21865e-011, 5.40797e-011},
			{-2.72655, 2.05133, 5.00011},
			{-5.44383, 4.10855, 10.0146},
			{-8.12646, 6.15252, 14.9968},
			{-7.04737, 5.26351, 12.8835},
			{-21.812, 16.3419, 40},
			{-6.55974, 4.93012, 12.0674},
			{-5.42734, 4.09191, 10.0158},
			{-4.38035, 3.25789, 8.00783},
			{-10.3607, 7.72991, 19},
			{-109.805, 82.1812, 202},
			{-55.2718, 41.4974, 102},
			{-45.9487, 34.0308, 84},
			{-159.678, 118.632, 292.827},
			{-158.926, 118.445, 292.364},
			{-158.677, 118.632, 292.827},
	
		};
		
		std::vector<Eigen::Vector3f> ngt {
			
			{0.585891, -0.58591, 0.559859},
			{0.421129, -0.735943, 0.530131},
			{0.740737, -0.379317, 0.554461},
			{0, 0, 0},
			{0.737575, 0.662444, 0.13096},
			{-0.750113, 0.358166, -0.55592},
			{-0.204899, -0.940479, 0.271139},
			{0, 0, 0},
			{0.454634, -0.710914, 0.536571},
			{0.577978, -0.596587, 0.556799},
			{-0.886194, -0.279371, -0.369609},
			{0, 0, 0},
			{0, 0, 0},
			{0, 0, 0},
			{0, 0, 0},
			{0, 0, 0}
			
		};


		pv.emplace(std::move(v));
		
		WHEN("It is invoked") {
			
			auto t=kfompb(pv, width, height, k);
			auto && v=std::get<0>(t);
			auto && n=std::get<1>(t);
			auto && vs=v->get();
			auto && ns=n->get();
			
			THEN("The returned normals are all either exactly 0 or close to 1 in length") {
				
				for(auto && n : ns) {
				
					CHECK((n.norm() == Approx( 1.0f ) || n.isZero()));
				
				}
				
				
			}
			
			THEN("The returned vertices match the ground truth vertices within 0.001") {
				
				// Note: Eigen's default precision was too precise, here we only care about generally close
				CHECK ( std::equal(vgt.begin(),vgt.end(),vs.begin(),vs.end(),[] (Eigen::Vector3f a, Eigen::Vector3f b) noexcept { 
				
					return (a-b).isZero(0.001f);
				
				}) );

				
			}
			
			THEN("The returned normals match the ground truth normals within 0.001") {
				
				// Note: Eigen's default precision was too precise, here we only care about generally close
				CHECK ( std::equal(ngt.begin(),ngt.end(),ns.begin(),ns.end(),[] (Eigen::Vector3f a, Eigen::Vector3f b) noexcept { 
				
					return (a-b).isZero(0.001f);
				
				}) );

				
			}
			
		}
		
	}
	
}
