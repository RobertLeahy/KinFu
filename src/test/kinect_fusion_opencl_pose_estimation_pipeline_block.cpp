#include <dynfu/kinect_fusion_opencl_pose_estimation_pipeline_block.hpp>


#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/file_system_depth_device.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>
#include <dynfu/msrc_file_system_depth_device.hpp>
#include <dynfu/path.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>
#include <catch.hpp>


namespace {


	class fixture {


		private:


			static dynfu::filesystem::path curr_dir () {

				return dynfu::filesystem::path(dynfu::current_executable_parent_path());

			}


			static dynfu::filesystem::path cl_path () {

				auto retr=curr_dir();
				retr/="..";
				retr/="cl";

				return retr;

			}


			static dynfu::filesystem::path frames_path () {

				auto retr=curr_dir();
				retr/="..";
				retr/="data/test/kinect_fusion_eigen_pose_estimation_pipeline_block";

				return retr;

			}


		public:


			boost::compute::device dev;
			boost::compute::context ctx;
			boost::compute::command_queue q;
			std::size_t width;
			std::size_t height;
			Eigen::Matrix3f k;
			Eigen::Matrix4f t_gk_initial;
			dynfu::file_system_opencl_program_factory pf;
			dynfu::msrc_file_system_depth_device_frame_factory ff;
			dynfu::msrc_file_system_depth_device_filter f;
			dynfu::file_system_depth_device dd;
			dynfu::kinect_fusion_opencl_measurement_pipeline_block mpb;


			fixture ()
				:	dev(boost::compute::system::default_device()),
					ctx(dev),
					q(ctx,dev),
					width(640),
					height(480),
					t_gk_initial(Eigen::Matrix4f::Identity()),
					pf(cl_path(),ctx),
					dd(frames_path(),ff,&f),
					mpb(q,pf,16,2.0f,1.0f)
			{

				k <<	585.0f, 0.0f, 320.0f,
						0.0f, -585.0f, 240.0f,
						0.0f, 0.0f, 1.0f;
				
				t_gk_initial(0,3)=1.5f;
				t_gk_initial(1,3)=1.5f;
				t_gk_initial(2,3)=1.5f;

			}


	};


}


SCENARIO_METHOD(fixture,"dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block objects return the matrix passed to their constructor on their first invocation","[dynfu][pose_estimation_pipeline_block][kinect_fusion_opencl_pose_estimation_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block object") {

		dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block pepb(pf,q,0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial);

		WHEN("It is invoked for the first time (i.e. passed NULL as the 3rd, 4th, and 5th arguments to dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block::operator ())") {

			auto frame_ptr=dd();
			auto t=mpb(*frame_ptr,width,height,k);
			auto && v=std::get<0>(t);
			auto && n=std::get<1>(t);
			auto ptr=pepb(*v,*n,nullptr,nullptr,k,{});

			THEN("The T_gk matrix provided to its constructor is returned") {

				CHECK(ptr->get()==t_gk_initial);

			}

		}

	}

}


SCENARIO_METHOD(fixture,"dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block objects derive a transformation matrix between two identical frames that is the identity matrix with some set, constant, initial translation","[dynfu][pose_estimation_pipeline_block][kinect_fusion_opencl_pose_estimation_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block object") {

		dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block pepb(pf,q,0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial);

		WHEN("It is invoked on two identical sets of vertices and normals") {

			auto frame_ptr=dd();
			auto t=mpb(*frame_ptr,width,height,k);
			auto && v=std::get<0>(t);
			auto && n=std::get<1>(t);
			auto ptr=pepb(*v,*n,nullptr,nullptr,k,{});
			ptr=pepb(*v,*n,v.get(),n.get(),k,std::move(ptr));

			THEN("The resulting T_gk matrix is an identity matrix with the translation components from the initial T_gk matrix") {

				auto t_gk=ptr->get();
				auto diff=t_gk-t_gk_initial;
				CHECK(diff.isZero());

			}

		}

	}

}
