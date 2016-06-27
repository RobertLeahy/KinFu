#include <dynfu/kinect_fusion_eigen_pose_estimation_pipeline_block.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/pipeline_value.hpp>
#include <cmath>
#include <exception>
#include <memory>
#include <tuple>
#include <utility>

namespace dynfu {

	static bool isfinite (const Eigen::Vector3f & v) noexcept {
		return std::isfinite(v(0)) && std::isfinite(v(1)) && std::isfinite(v(2));
	}

	kinect_fusion_eigen_pose_estimation_pipeline_block::kinect_fusion_eigen_pose_estimation_pipeline_block (
		float epsilon_d,
		float epsilon_theta,
		std::size_t frame_width, 
		std::size_t frame_height,
		std::size_t numit
	) : epsilon_d_(epsilon_d), epsilon_theta_(epsilon_theta), frame_width_(frame_width), frame_height_(frame_height), numit_(numit) {}


	Eigen::Matrix4f kinect_fusion_eigen_pose_estimation_pipeline_block::iterate(
		measurement_pipeline_block::value_type & live_vn,
		surface_prediction_pipeline_block::value_type & predicted_previous_vn,
		Eigen::Matrix3f k,
		Eigen::Matrix4f t_frame_frame,
		Eigen::Matrix4f t_z
	) {

		// Compute correspondences via projective data association

		std::size_t rej_m = 0;
		std::size_t rej_ed = 0;
		std::size_t rej_et = 0;
		std::size_t numpts = frame_height_ * frame_width_;

		std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>> correspondences;

		for (std::size_t i = 0; i < numpts; ++i) {

			auto v = std::get<0>(live_vn)->get()[i];
			auto n = std::get<1>(live_vn)->get()[i];

			Eigen::Vector4f v_homo(v(0), v(1), v(2), 1.0f);

			auto persp = t_frame_frame * v_homo;

			Eigen::Vector3f persp_div(persp(0)/persp(3), persp(1)/persp(3), persp(2)/persp(3));

			auto image_plane = k * persp_div;
			Eigen::Vector2d u( std::round(image_plane(0)/image_plane(2)), std::round(image_plane(1)/image_plane(2)));

			// stored row maj
			std::size_t lin_idx = u(0) + u(1) * frame_width_;

			if (lin_idx >= numpts) continue;

			auto pv = std::get<0>(predicted_previous_vn)->get()[lin_idx];
			auto pn = std::get<1>(predicted_previous_vn)->get()[lin_idx];
			
			// Test correspondences rejection critea (Eqn 17)
			
			// TODO: use M_k instead of this?
			if (!isfinite(pn) | !isfinite(pv) | !isfinite(v)) {
				rej_m++;
				continue;
			}

			// Test epsilon d
			Eigen::Vector4f pv_homo(pv(0), pv(1), pv(2), 1.0);
			if ( (t_z * v_homo - pv_homo).norm() > epsilon_d_) {
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
				v,
				pv,
				pn	 
			));

		}

		// TODO: What percent to actually use
		if (correspondences.size() < 0.1*frame_width_*frame_height_) {
			// I think this means we have blown our tracking... what to do...
			throw std::runtime_error("Less than 10% of correspondences were accepted, tracking lost");
		} 

		// build and sum up the matrices
		Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
		Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 1, 6>::Zero();

		for (auto && cor : correspondences) {
		
			auto && p = std::get<0>(cor);
			auto && q = std::get<1>(cor);
			auto && n = std::get<2>(cor);

			auto c = p.cross(n);

			auto pqn = (p-q).dot(n);

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
		
		Eigen::Matrix4f to_return;
		to_return << 1.0f, x(0), -x(2), x(3),
					 -x(0), 1.0f, x(1), x(4),
					 x(2), -x(1), 1.0f, x(5),
					 0.0f, 0.0f, 0.0f, 1.0f;
		
		
		return to_return;

	}

	kinect_fusion_eigen_pose_estimation_pipeline_block::value_type kinect_fusion_eigen_pose_estimation_pipeline_block::operator () (
		measurement_pipeline_block::value_type & live_vn,
		surface_prediction_pipeline_block::value_type & predicted_previous_vn,
		Eigen::Matrix3f k,
		Eigen::Matrix4f t_g_k_minus_one
	) {

		Eigen::Matrix4f t_frame_frame = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f t_z = t_g_k_minus_one;
		
		for (std::size_t i = 0; i < numit_; ++i) {
			t_z = kinect_fusion_eigen_pose_estimation_pipeline_block::iterate(live_vn, predicted_previous_vn, k, t_frame_frame, t_z);
			t_frame_frame = t_g_k_minus_one.inverse() * t_z; // TODO: is this really the inverse?
		}
		
		auto ptr = std::make_unique<cpu_pipeline_value<sensor_pose_estimation_type>>();
		ptr->emplace(t_z);
		return std::move(ptr);
	}
	


}
