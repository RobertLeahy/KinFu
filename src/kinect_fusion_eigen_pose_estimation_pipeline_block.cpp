#include <dynfu/kinect_fusion_eigen_pose_estimation_pipeline_block.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/pipeline_value.hpp>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>


#include <iostream>

namespace dynfu {

	static bool isfinite (const Eigen::Vector3f & v) noexcept {
		return std::isfinite(v(0)) && std::isfinite(v(1)) && std::isfinite(v(2));
	}
	kinect_fusion_eigen_pose_estimation_pipeline_block::kinect_fusion_eigen_pose_estimation_pipeline_block (
		float epsilon_d,
		float epsilon_theta,
		std::size_t frame_width, 
		std::size_t frame_height,
		Eigen::Matrix4f t_gk_initial,
		std::size_t numit
	) : epsilon_d_(epsilon_d), epsilon_theta_(epsilon_theta), frame_width_(frame_width), frame_height_(frame_height), numit_(numit), t_gk_initial_(std::move(t_gk_initial)) {}


	Eigen::Matrix4f kinect_fusion_eigen_pose_estimation_pipeline_block::iterate(
		measurement_pipeline_block::vertex_value_type::element_type & v_i,
		measurement_pipeline_block::normal_value_type::element_type & n_i,
		measurement_pipeline_block::vertex_value_type::element_type * prev_v,
		measurement_pipeline_block::normal_value_type::element_type * prev_n,
		Eigen::Matrix3f k,
		Eigen::Matrix4f t_frame_to_frame,
		Eigen::Matrix4f t_z
	) {

		// Compute correspondences via projective data association

		std::size_t rej_m = 0;
		std::size_t rej_ed = 0;
		std::size_t rej_et = 0;
		std::size_t rej_oob = 0;
		std::size_t numpts = frame_height_ * frame_width_;

		std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>> correspondences;

		for (std::size_t i = 0; i < numpts; ++i) {

			// measured v and n maps from current frame. These are in current camera space.
			auto v = v_i.get()[i];
			auto n = n_i.get()[i];
			Eigen::Vector4f v_h(v(0), v(1), v(2), 1.0f);

			// Transform v to previous camera space, using t_frame_to_frame
			auto v_cp_h = (t_frame_to_frame * v_h).eval();

			assert(v_cp_h(3) == 1.0f);

			// Convert to image coordinates
			Eigen::Vector3f v_cp(v_cp_h(0), v_cp_h(1), v_cp_h(2));
			auto uv3 = k * v_cp;
			Eigen::Vector2f uv(uv3(0)/uv3(2),uv3(1)/uv3(2));

			// convert uv coords to lin_idx
			std::int32_t lin_idx = std::int32_t(std::round(uv(0))) + std::int32_t(std::round(uv(1))) * frame_width_;

			if (lin_idx >= std::int32_t(numpts) || lin_idx < 0)  {
				rej_oob++;
				continue;
			}

			// In world coordinates
			auto pv = prev_v->get()[lin_idx];
			auto pn = prev_n->get()[lin_idx];

			if (!isfinite(pn) || !isfinite(pv) || !isfinite(v)) {
				rej_m++;
				continue;
			}



			// Test epsilon d
			Eigen::Vector4f pv_homo(pv(0), pv(1), pv(2), 1.0);
			Eigen::Vector4f v_w_h = t_z * v_h;
			assert(v_w_h(3) == 1.0f);
			if ( (v_w_h - pv_homo).norm() > epsilon_d_) {
				rej_ed++;
				continue;
			} 

			// Test epsilon theta
			auto r_z = t_z.block<3,3>(0,0); // rotation block of t_z
			if ( (r_z*n).cross(pn).norm() > epsilon_theta_ ) {
				rej_et++;
				continue;
			}
			
			// if we get to here the correspondence is valid
			correspondences.emplace_back(std::make_tuple(
				v, // <- in current camera space
				pv,
				pn	 
			));
		}

		// 6 is the DOF, so we need 6 points to solve minimum
		if (correspondences.size() < 6) {
			// I think this means we have blown our tracking... what to do...
			throw pose_estimation_pipeline_block::tracking_lost_error("Less than 6 correspondences were found");
		} 

		// build and sum up the matrices
		Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
		Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 1, 6>::Zero();

		for (auto && cor : correspondences) {
		
			auto && p = std::get<0>(cor); // v
			auto && q = std::get<1>(cor); // pv
			auto && n = std::get<2>(cor); // pn

			auto c = p.cross(n);

			auto pqn = (p-q).dot(n); // for b vector

			Eigen::Matrix<float, 6, 6> Ai;
			Ai << c(0)*c(0), c(0)*c(1), c(0)*c(2), c(0)*n(0), c(0)*n(1), c(0)*n(2),
				  c(1)*c(0), c(1)*c(1), c(1)*c(2), c(1)*n(0), c(1)*n(1), c(1)*n(2),
				  c(2)*c(0), c(2)*c(1), c(2)*c(2), c(2)*n(0), c(2)*n(1), c(2)*n(2),
				  n(0)*c(0), n(0)*c(1), n(0)*c(2), n(0)*n(0), n(0)*n(1), n(0)*n(2),
				  n(1)*c(0), n(1)*c(1), n(1)*c(2), n(1)*n(0), n(1)*n(1), n(1)*n(2),
				  n(2)*c(0), n(2)*c(1), n(2)*c(2), n(2)*n(0), n(2)*n(1), n(2)*n(2);

			
			Eigen::Matrix<float, 6, 1> bi;
			
			bi << c(0), c(1), c(2), n(0), n(1), n(2);
			bi *= pqn * (-1);

			A = A + Ai;
			b = b + bi;

		}


		// Solve the system by column pivot - TODO: use Cholesky?
		Eigen::Matrix<float, 6, 1> x = A.colPivHouseholderQr().solve(b);
		
		float alpha = x(0);
		float beta =  x(1);
		float gamma = x(2);
		float tx = x(3);
		float ty = x(4);
		float tz = x(5);

		Eigen::Matrix4f to_return;
		to_return << 1.0f, -gamma, beta, tx,
					 gamma, 1.0f, -alpha, ty,
				 	 -beta, alpha, 1.0f, tz,
					 0.0f, 0.0f, 0.0f, 1.0f;
		

		return to_return;

	}

	kinect_fusion_eigen_pose_estimation_pipeline_block::value_type kinect_fusion_eigen_pose_estimation_pipeline_block::operator () (
		measurement_pipeline_block::vertex_value_type::element_type & v,
		measurement_pipeline_block::normal_value_type::element_type & n,
		measurement_pipeline_block::vertex_value_type::element_type * prev_v,
		measurement_pipeline_block::normal_value_type::element_type * prev_n,
		Eigen::Matrix3f k,
		value_type t_gk_minus_one
	) {
		
		using pv_type=cpu_pipeline_value<sensor_pose_estimation_type>;
		if (!(prev_v && prev_n && t_gk_minus_one)) {
			t_gk_minus_one=std::make_unique<pv_type>();
			static_cast<pv_type &>(*t_gk_minus_one).emplace(t_gk_initial_);
			return t_gk_minus_one;
		}

		Eigen::Matrix4f t_frame_to_frame = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f t_z = t_gk_minus_one->get(); // z= 0 means t_z = T_g_k from previous frame

		for (std::size_t i = 0; i < numit_; ++i) {
			t_z = kinect_fusion_eigen_pose_estimation_pipeline_block::iterate(v, n, prev_v, prev_n, k, t_frame_to_frame, t_z);
			std::cout << "T_z" << std::endl << t_z << std::endl;
			t_frame_to_frame = t_gk_minus_one->get().inverse() * t_z;
		}

		auto && pv=dynamic_cast<cpu_pipeline_value<sensor_pose_estimation_type> &>(*t_gk_minus_one);
		pv.emplace(t_z);
		return t_gk_minus_one;
	}
	


}
