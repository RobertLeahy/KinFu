#include <dynfu/kinect_fusion_eigen_pose_estimation_pipeline_block.hpp>

#include <boost/compute.hpp>

#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/file_system_depth_device.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>
#include <dynfu/msrc_file_system_depth_device.hpp>
#include <dynfu/opencl_depth_device.hpp>
#include <dynfu/path.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include <catch.hpp>

namespace {


	class fixture {
		
			private:
			
				static dynfu::filesystem::path cl_path () {
					
					dynfu::filesystem::path retr(dynfu::current_executable_parent_path());
					retr/="..";
					retr/="cl";
					
					return retr;
					
				}
		
			protected:
				

				boost::compute::device dev;
				boost::compute::context ctx;
				boost::compute::command_queue q;
				std::size_t width;
				std::size_t height;
				dynfu::measurement_pipeline_block::value_type live_vn_;
				dynfu::surface_prediction_pipeline_block::value_type prev_vn_;
				Eigen::Matrix3f k;
				Eigen::Matrix4f t_gk_minus_one;
				dynfu::file_system_opencl_program_factory fsopf;
				
				
			public:
				
				fixture(): dev(boost::compute::system::default_device()), ctx(dev), q(ctx,dev), width(640), height(480), t_gk_minus_one(Eigen::Matrix4f::Identity()), fsopf(cl_path(),ctx){
					
					k <<  585.0f, 0.0f, 320.0f,
					 0.0f, -585.0f, 240.0f,
					 0.0f, 0.0f, 1.0f;
					
					
				}
		
	};


}



SCENARIO_METHOD(fixture, "A kinect_fusion_eigen_pose_estimation_pipeline_block uses Eigen to find an updated sensor pose estimation", "[dynfu][pipeline_block][pose_estimation_pipeline_block][kinect_fusion_eigen_pose_estimation_pipeline_block][!hide]") {
	
	GIVEN("A kinect_fusion_eigen_pose_estimation_pipeline_block") {
		
		dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block kfepeb(0.10f, std::sin(20.0f * 3.14159f / 180.0f), 640, 480); // took default params from KinFu
		dynfu::kinect_fusion_opencl_measurement_pipeline_block kfompb(q, fsopf, 16, 2.0f, 1.0f);
		dynfu::kinect_fusion_opencl_measurement_pipeline_block kfompb2(q, fsopf, 16, 2.0f, 1.0f);

		
		WHEN("It is invoked on a set of identical vertices and normals") {
			
			dynfu::cpu_pipeline_value<std::vector<float>> pv;
			dynfu::cpu_pipeline_value<std::vector<float>> pv2;
			
			dynfu::filesystem::path pp(dynfu::current_executable_parent_path());

			
			dynfu::msrc_file_system_depth_device_frame_factory ff;
			dynfu::msrc_file_system_depth_device_filter f;
			dynfu::file_system_depth_device ddi(pp/".."/"data/test/msrc_file_system_depth_device",ff,&f);
			dynfu::opencl_depth_device dd(ddi,q);

			auto v=dd()->get(); // frame 00
			pv.emplace(v);
			pv2.emplace(std::move(v));
			 
			auto t=kfompb(pv, width, height, k);
			auto t2=kfompb2(pv2, width, height, k);
			
			
			THEN("The determined t_g_k matrix is an identity matrix.") {
				
					auto && s = kfepeb(t, t2, k, t_gk_minus_one);
					
					Eigen::Matrix4f t_k = s->get();
					
					CHECK((t_k - Eigen::Matrix4f::Identity()).isZero());

				
			}
			
		}

		 WHEN("It is invoked on a set of consecutive frames") {
			
			dynfu::cpu_pipeline_value<std::vector<float>> pv;
			dynfu::cpu_pipeline_value<std::vector<float>> pv2;
			
			dynfu::filesystem::path pp(dynfu::current_executable_parent_path());

			
			dynfu::msrc_file_system_depth_device_frame_factory ff;
			dynfu::msrc_file_system_depth_device_filter f;
			dynfu::file_system_depth_device ddi(pp/".."/"data/test/kinect_fusion_eigen_pose_estimation_pipeline_block",ff,&f);
			dynfu::opencl_depth_device dd(ddi,q);
		
			
			pv.emplace(std::move(dd()->get())); // frame 00
			pv2.emplace(std::move(dd()->get())); // frame 01
			
 
			auto t=kfompb(pv, width, height, k);
			auto t2=kfompb2(pv2, width, height, k);
			
			
			THEN("The determined t_g_k matrix is valid") {
				
					auto && s = kfepeb(t, t2, k, t_gk_minus_one);
					
					Eigen::Matrix4f t_k = s->get();
					
					Eigen::Matrix4f gt;
					gt <<  1.0f,  -0.00118862f,  -0.00169119f,	 -0.0011603f,
						   0.00118862f,			  1.0f,  0.000905547f,	 0.00040294f,
						   0.00169119f, -0.000905547f,			 1.0f, -0.000257364f,
						   0.0f,			0.0f,			0.0f,			1.0f;
					
					CHECK((t_k - gt).isZero());

				
			}
		}

		WHEN("It is invoked on a set of non-consecutive frames") {
		
			dynfu::cpu_pipeline_value<std::vector<float>> pv;
			dynfu::cpu_pipeline_value<std::vector<float>> pv2;
			
			dynfu::filesystem::path pp(dynfu::current_executable_parent_path());

			
			dynfu::msrc_file_system_depth_device_frame_factory ff;
			dynfu::msrc_file_system_depth_device_filter f;
			dynfu::file_system_depth_device ddi(pp/".."/"data/test/kinect_fusion_eigen_pose_estimation_pipeline_block",ff,&f);
			dynfu::opencl_depth_device dd(ddi,q);
		
			
			pv.emplace(std::move(dd()->get())); // frame 00
			dd(); // frame 01
			pv2.emplace(std::move(dd()->get())); // frame 23
			

			auto t=kfompb(pv, width, height, k);
			auto t2=kfompb2(pv2, width, height, k);
			
			
			THEN("A tracking failure occurs") {
				
				CHECK_THROWS_AS(kfepeb(t, t2, k, t_gk_minus_one), dynfu::pose_estimation_pipeline_block::tracking_lost_error);
					
				
			}
		}
		
	}
	
}
