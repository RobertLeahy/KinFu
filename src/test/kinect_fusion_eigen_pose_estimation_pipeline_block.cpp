/**
 * Note that these tests are NOT run by default. To run these: bin\tests [kinect_fusion_eigen_pose_estimation_pipeline_block]
 */
#include <dynfu/kinect_fusion_eigen_pose_estimation_pipeline_block.hpp>


#include <boost/compute.hpp>
#include <boost/nondet_random.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/file_system_depth_device.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>
#include <dynfu/msrc_file_system_depth_device.hpp>
#include <dynfu/path.hpp>
#include <dynfu/pixel.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <random>
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
			boost::random_device rd;
			std::mt19937 mt;
			std::uniform_real_distribution<float> angle_dist;
			std::uniform_real_distribution<float> trans_dist;


			std::vector<dynfu::pixel> to_global (std::vector<dynfu::pixel> ps) {

				std::transform(ps.begin(),ps.end(),ps.begin(),[&] (const auto & p) noexcept {

					Eigen::Vector4f vh(p.v(0),p.v(1),p.v(2),1.0f);
					vh=t_gk_initial*vh;
					Eigen::Vector3f v(vh(0),vh(1),vh(2));

					return dynfu::pixel{v,t_gk_initial.block<3,3>(0,0)*p.n};

				});

				return ps;

			}

			std::vector<dynfu::pixel> perturb (std::vector<dynfu::pixel> ps, Eigen::Matrix4f transform) {

				std::transform(ps.begin(),ps.end(),ps.begin(),[&] (const auto & p) noexcept {

					Eigen::Vector4f vh(p.v(0),p.v(1),p.v(2),1.0f);
					vh=transform*vh;
					Eigen::Vector3f v(vh(0),vh(1),vh(2));

					return dynfu::pixel{v,transform.block<3,3>(0,0)*p.n};

				});

				return ps;

			}

			Eigen::Matrix4f rand_perturbation() {

				// generate 3 random angles
				auto alpha = angle_dist(mt);
				auto beta = angle_dist(mt);
				auto gamma = angle_dist(mt);

				// generate 3 random translation
				auto dx = trans_dist(mt);
				auto dy = trans_dist(mt);
				auto dz = trans_dist(mt);


				Eigen::Vector3f dt(dx,dy,dz);
				dt = dt.normalized() * trans_dist(mt); // scale to 1 cm max distance

				Eigen::Matrix4f retr;
				retr <<  1.0f,  alpha, beta, dt(0),
								-alpha, 1.0f, gamma, dt(1),
								-beta, -gamma, 1.0f, dt(2),
								0.0f,  0.0f,  0.0f,  1.0f;
				
				return retr;

			}


			fixture ()
				:	dev(boost::compute::system::default_device()),
					ctx(dev),
					q(ctx,dev),
					width(640),
					height(480),
					t_gk_initial(Eigen::Matrix4f::Identity()),
					pf(cl_path(),ctx),
					dd(frames_path(),ff,&f),
					mpb(q,pf,16,2.0f,1.0f),
					rd(),
					mt(rd),
					angle_dist(-0.02, 0.02),
					trans_dist(-0.01, 0.01)

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


SCENARIO_METHOD(fixture,"dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block objects return the matrix passed to their constructor on their first invocation","[!hide][dynfu][pose_estimation_pipeline_block][kinect_fusion_eigen_pose_estimation_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block object") {

		dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block pepb(0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial);

		WHEN("It is invoked for the first time (i.e. passed NULL as the 3rd, 4th, and 5th arguments to dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block::operator ())") {

			auto frame_ptr=dd();
			auto m_ptr=mpb(*frame_ptr,width,height,k);
			auto ptr=pepb(*m_ptr,nullptr,k,{});

			THEN("The T_gk matrix provided to its constructor is returned") {

				CHECK(ptr->get()==t_gk_initial);

			}

		}

	}

}


SCENARIO_METHOD(fixture,"dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block objects derive a transformation matrix between two identical frames that is the identity matrix with some set, constant, initial translation","[!hide][dynfu][pose_estimation_pipeline_block][kinect_fusion_eigen_pose_estimation_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block object") {

		dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block pepb(0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial);

		WHEN("It is invoked on two identical sets of vertices and normals") {

			auto frame_ptr=dd();
			auto m_ptr=mpb(*frame_ptr,width,height,k);
			auto ptr=pepb(*m_ptr,nullptr,k,{});
			dynfu::cpu_pipeline_value<std::vector<dynfu::pixel>> pv;
			pv.emplace(to_global(m_ptr->get()));
			ptr=pepb(*m_ptr,&pv,k,std::move(ptr));

			THEN("The resulting T_gk matrix is an identity matrix with the translation components from the initial T_gk matrix") {

				auto t_gk=ptr->get();
				auto diff=t_gk-t_gk_initial;
				CHECK(diff.isZero());

			}

		}

	}


}

SCENARIO_METHOD(fixture,"dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block objects derive a transformation matrix between consecutive MSRC frames that matches the expected value, when forcing pixel to pixel correspondences","[!hide][dynfu][pose_estimation_pipeline_block][kinect_fusion_eigen_pose_estimation_pipeline_block][!mayfail]") {

	GIVEN("A dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block object with force pixel to pixel correspondences enabled") {

		bool enable_pixel_to_pixel(false);
		dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block pepb(0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial, 15, enable_pixel_to_pixel);

		WHEN("It is invoked on consecutive MSRC frames") {

			auto frame_ptr=dd();
			auto m_ptr=mpb(*frame_ptr,width,height,k);
			auto ptr=pepb(*m_ptr,nullptr,k,{});

			auto frame_ptr2=dd();
			auto m_ptr2=mpb(*frame_ptr2,width,height,k);

			dynfu::cpu_pipeline_value<std::vector<dynfu::pixel>> pv;
			pv.emplace(to_global(m_ptr->get()));
			ptr=pepb(*m_ptr2,&pv,k,std::move(ptr));

			Eigen::Matrix4f gt;
			gt << 0.999998443679553f,  -0.001401553458469f,  -0.001091006688159f,   1.496718902792527f,
					0.001403957028225f,   0.999996648369455f,   0.002197726537721f,   1.505616570079536f,
					0.001087974352801f,  -0.002199269742567f,   0.999996929166082f,   1.498810789733843f,
					0.0f,                   0.0f,                   0.0f,   1.0f;


			THEN("The kinect_fusion_eigen_pose_estimation_pipeline_block finds the correct transformation") {
			

				auto t_gk=ptr->get(); // what we get back (current cam -> world)

				CHECK((t_gk-gt).isZero());

				

			}


		
		}

	}


}

SCENARIO_METHOD(fixture,"dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block objects derive a transformation matrix between a frame and its perturbed version that is within numerical precision of the perturbation","[!hide][dynfu][pose_estimation_pipeline_block][kinect_fusion_eigen_pose_estimation_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block object") {

		bool enable_pixel_to_pixel(false);
		dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block pepb(0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial,15,enable_pixel_to_pixel);

		WHEN("It is invoked on a set of vertices and normals, and a set of perturbed vertices and normals") {

			auto frame_ptr=dd();
			auto m_ptr=mpb(*frame_ptr,width,height,k);
			auto ptr=pepb(*m_ptr,nullptr,k,{});

			auto perturbation = rand_perturbation();

			auto gt = perturbation * t_gk_initial; // should also be from current cam -> world

			dynfu::cpu_pipeline_value<std::vector<dynfu::pixel>> pv;
			pv.emplace(perturb(to_global(m_ptr->get()), perturbation));
			ptr=pepb(*m_ptr,&pv,k,std::move(ptr));


			THEN("The kinect_fusion_eigen_pose_estimation_pipeline_block finds the applied perturbation.") {
			
				auto t_gk=ptr->get(); // what we get back (current cam -> world)

				CHECK((t_gk-gt).isZero());

				

			}


		
		}

	}


}



static bool is_bounded_by (float val, float a, float b) noexcept {

	if (val>std::max(a,b)) return false;
	if (val<std::min(a,b)) return false;
	return true;

}


SCENARIO_METHOD(fixture,"Between consecutive frames dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block reports only a small change in sensor pose","[!hide][dynfu][kinect_fusion_eigen_pose_estimation_pipeline_block][pose_estimation_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block object") {



		dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block pepb(0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial);

		WHEN("It is invoked on two consecutive frames") {

			auto frame_ptr=dd();
			auto m_ptr1=mpb(*frame_ptr,width,height,k);
			auto ptr=pepb(*m_ptr1,nullptr,k,{});
			frame_ptr=dd(std::move(frame_ptr));
			auto m_ptr2=mpb(*frame_ptr,width,height,k);
			dynfu::cpu_pipeline_value<std::vector<dynfu::pixel>> pv;
			pv.emplace(to_global(m_ptr1->get()));
			ptr=pepb(*m_ptr2,&pv,k,std::move(ptr));

			THEN("The resulting T_gk matrix is not simply the initial T_gk matrix") {

				CHECK(ptr->get()!=t_gk_initial);

			}

			THEN("The resulting T_gk matrix indicates a small change in camera pose") {

				auto t_gk=ptr->get();

				//	There should be minimal translation
				float min=1.49f;
				float max=1.51f;
				CHECK(is_bounded_by(t_gk(0,3),min,max));
				CHECK(is_bounded_by(t_gk(1,3),min,max));
				CHECK(is_bounded_by(t_gk(2,3),min,max));

				//	Bottom row should be == [0 0 0 1]
				CHECK(t_gk(3,0)==Approx(0.0f));
				CHECK(t_gk(3,1)==Approx(0.0f));
				CHECK(t_gk(3,2)==Approx(0.0f));
				CHECK(t_gk(3,3)==Approx(1.0f));

				//	The 3x3 upper left block should be a
				//	rotation matrix (there should be no scaling)
				//
				//	Note: There are matrix entries that are never
				//	accessed, we have a check for that later
				//
				//	Formulae from here: http://nghiaho.com/?page_id=846
				auto x_rot=std::atan2(t_gk(2,1),t_gk(2,2));
				auto y_rot=std::atan2(-t_gk(2,0),std::sqrt(std::pow(t_gk(2,1),2)+std::pow(t_gk(2,2),2)));
				auto z_rot=std::atan2(t_gk(1,0),t_gk(0,0));
				//	We expect small angles, say 5 degrees or less
				auto five_degrees=0.0872665f;
				CHECK(is_bounded_by(x_rot,five_degrees,-five_degrees));
				CHECK(is_bounded_by(y_rot,five_degrees,-five_degrees));
				CHECK(is_bounded_by(z_rot,five_degrees,-five_degrees));
				//	Now we recreate the rotation matrix and make
				//	sure it's approximately the same, this is how
				//	we check the entries we didn't touch above
				Eigen::Matrix3f x;
				x <<	1.0f, 0.0f, 0.0f,
						0.0f, std::cos(x_rot), -std::sin(x_rot),
						0.0f, std::sin(x_rot), std::cos(x_rot);
				Eigen::Matrix3f y;
				y <<	std::cos(y_rot), 0.0f, std::sin(y_rot),
						0.0f, 1.0f, 0.0f,
						-std::sin(y_rot), 0.0f, std::cos(y_rot);
				Eigen::Matrix3f z;
				z << 	std::cos(z_rot), -std::sin(z_rot), 0.0f,
						std::sin(z_rot), std::cos(z_rot), 0.0f,
						0.0f, 0.0f, 1.0f;
				Eigen::Matrix3f rot(z*y*x);
				Eigen::Matrix3f diff(t_gk.block<3,3>(0,0)-rot);
				CHECK(diff.isZero());

			}

		}

	}

}


SCENARIO_METHOD(fixture,"When a minimum number of point-to-point correspondences cannot be made dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block objects throw a dynfu::pose_estimation_pipeline_block::tracking_lost_error","[!hide][dynfu][kinect_fusion_eigen_pose_estimation_pipeline_block][pose_estimation_pipeline_block][!mayfail]") {

	GIVEN("A dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block object") {

		dynfu::kinect_fusion_eigen_pose_estimation_pipeline_block pepb(0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial);

		THEN("Invoking it on two non-consecutive frames results in a dynfu::pose_estimation_pipeline_block::tracking_lost_error") {

			auto frame_ptr=dd();	//	Frame 00
			auto m_ptr1=mpb(*frame_ptr,width,height,k);
			frame_ptr=dd(std::move(frame_ptr));	//	Frame 01
			frame_ptr=dd(std::move(frame_ptr));	//	Frame 23
			auto m_ptr2=mpb(*frame_ptr,width,height,k);
			auto ptr=pepb(*m_ptr1,nullptr,k,{});
			dynfu::cpu_pipeline_value<std::vector<dynfu::pixel>> pv;
			pv.emplace(to_global(m_ptr2->get()));
			CHECK_THROWS_AS(pepb(*m_ptr1,&pv,k,std::move(ptr)),dynfu::pose_estimation_pipeline_block::tracking_lost_error);
			
		}

	}

}
