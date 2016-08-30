#include <kinfu/kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block.hpp>


#include <boost/compute.hpp>
#include <kinfu/cpu_pipeline_value.hpp>
#include <kinfu/filesystem.hpp>
#include <kinfu/half.hpp>
#include <kinfu/path.hpp>
#include <kinfu/pixel.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <tuple>
#include <vector>
#include <catch.hpp>


namespace {


	class fixture {

		private:

			static kinfu::filesystem::path cl_path () {

				kinfu::filesystem::path retr(kinfu::current_executable_parent_path());
				retr/="..";
				retr/="cl";

				return retr;

			}

		protected:

			boost::compute::device dev;
			boost::compute::context ctx;
			boost::compute::command_queue q;

		public:

		fixture () : dev(boost::compute::system::default_device()), ctx(dev), q(ctx,dev) {	}

	};


}


SCENARIO_METHOD(fixture,"kinfu::kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block objects simply copy the provided vertex and normal map to their output","[kinfu][surface_prediction_pipeline_block][kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block]") {

	GIVEN("A kinfu::kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block object") {

		kinfu::kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block sppb(q);

		WHEN("It is invoked") {

			kinfu::cpu_pipeline_value<std::vector<kinfu::half>> tsdf;
			tsdf.emplace();
			kinfu::cpu_pipeline_value<Eigen::Matrix4f> pose;
			pose.emplace(Eigen::Matrix4f::Zero());
			kinfu::cpu_pipeline_value<std::vector<kinfu::pixel>> map_pv;
			auto && map=map_pv.emplace();
			map.push_back({{1.0f,2.0f,3.0f},{4.0f,5.0f,6.0f}});

			auto ptr=sppb(tsdf,0,0,0,pose,Eigen::Matrix3f::Zero(),map_pv,kinfu::kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block::value_type{});

			THEN("The vertex and normal map provided is simply copied into the result") {

				auto && result=ptr->get();
				CHECK(std::equal(result.begin(),result.end(),map.begin(),map.end(),[&] (const auto & a, const auto & b) noexcept {

					return (a.v==b.v) && (a.n==b.n);

				}));

			}

		}

	}

}
