#include <dynfu/kinect_fusion_opencl_surface_prediction_pipeline_block.hpp>
#include <dynfu/kinect_fusion_opencl_update_reconstruction_pipeline_block.hpp>
#include <dynfu/msrc_file_system_depth_device.hpp>
#include <dynfu/file_system_depth_device.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/opencl_depth_device.hpp>
#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/path.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include <igl/viewer/Viewer.h>

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
			Eigen::Matrix3f k;
			Eigen::Matrix4f t_g_k;
			dynfu::file_system_opencl_program_factory fsopf;
			std::size_t tsdf_width;
			std::size_t tsdf_height;
			std::size_t tsdf_depth;

		public:

		fixture () : dev(boost::compute::system::default_device()), ctx(dev), q(ctx,dev), width(640), height(480), fsopf(cl_path(),ctx), tsdf_width(256), tsdf_height(256), tsdf_depth(256) {

				//needs to be a valid k matrix for a meaningful test
				k << 585.0f, 0.0f, 320.0f,
					 0.0f, 585.0f, 240.0f,
					 0.0f, 0.0f, 1.0f;

				//needs to be a valid t_g_k for meaningful test (?)
				t_g_k = Eigen::Matrix4f::Identity();
				t_g_k(0,3) = 1.5f;
				t_g_k(1,3) = 1.5f;
				t_g_k(2,3) = 1.5f;


			}

	};


}


SCENARIO_METHOD(fixture, "A dynfu::kinect_fusion_opencl_surface_prediction_pipeline_block implements the surface_prediction phase of the kinect fusion pipeline on the GPU using OpenCL","[dynfu][surface_prediction_pipeline_block][kinect_fusion_opencl_surface_prediction_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_opencl_ruface_prediction_pipeline_block") {

		float mu = 5.0;

		dynfu::kinect_fusion_opencl_update_reconstruction_pipeline_block kfourpb(q, fsopf, mu, tsdf_width, tsdf_height, tsdf_depth);
		dynfu::kinect_fusion_opencl_surface_prediction_pipeline_block kfosppb(q, fsopf, mu, tsdf_width);
		dynfu::cpu_pipeline_value<std::vector<float>> pv;

		dynfu::filesystem::path pp(dynfu::current_executable_parent_path());

		auto d=boost::compute::system::default_device();
		boost::compute::context ctx(d);
		boost::compute::command_queue q(ctx,d);

		dynfu::msrc_file_system_depth_device_frame_factory ff;
		dynfu::msrc_file_system_depth_device_filter f;
		dynfu::file_system_depth_device ddi(pp/".."/"data/test/tsdf_viewer/",ff,&f);
		dynfu::opencl_depth_device dd(ddi,q);

		auto && tsdf=kfourpb(*dd(), width, height, k, t_g_k);

		THEN("Invoking it and downloading the produced V and N maps does not fail") {

			auto f = [&](){


					auto && vnmap = kfosppb(std::move(tsdf), k, t_g_k);
					auto && v=std::get<0>(vnmap);
					auto && n=std::get<1>(vnmap);
					auto && vs=v->get();
					auto && ns=n->get();

					std::cout << std::count_if(vs.begin(),vs.end(),[] (const auto & v) {

						return std::isnan(v(0));

					}) << std::endl;

					cv::Mat m(480,640,CV_32FC3);
					std::size_t idx(0);
					for (std::size_t i=0;i<480;++i) {

						for (std::size_t j=0;j<640;++j,++idx) {

							auto && v=vs[idx];
							if (idx<100) std::cout << "[" << v(0) << ", " << v(1) << ", " << v(2) << "]" << std::endl;
							m.at<Eigen::Vector3f>(cv::Point(j,i))=ns[idx].cwiseAbs();

						}

					}

					cv::imshow("depth",m);
					cv::waitKey();

			};

			CHECK_NOTHROW(f());

		}

	}

}
