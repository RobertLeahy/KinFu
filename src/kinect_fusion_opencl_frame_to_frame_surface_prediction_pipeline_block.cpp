#include <boost/compute.hpp>
#include <dynfu/boost_compute_detail_type_name_trait.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <dynfu/kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block.hpp>
#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <utility>


namespace dynfu {


	kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block::kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block (boost::compute::command_queue q) : q_(std::move(q)) {	}


	kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block::value_type kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block::operator () (
		update_reconstruction_pipeline_block::value_type::element_type &,
		std::size_t,
		std::size_t,
		std::size_t,
		pose_estimation_pipeline_block::value_type::element_type &,
		Eigen::Matrix3f,
		measurement_pipeline_block::vertex_value_type::element_type & v,
		measurement_pipeline_block::normal_value_type::element_type & n,
		value_type vn
	) {

		auto && prev_v_ptr=std::get<0>(vn);
		using v_type=opencl_vector_pipeline_value<Eigen::Vector3f>;
		if (!prev_v_ptr) prev_v_ptr=std::make_unique<v_type>(q_);
		auto && prev_v=dynamic_cast<v_type &>(*prev_v_ptr);
		auto && prev_v_vec=prev_v.vector();
		auto && prev_n_ptr=std::get<1>(vn);
		using n_type=opencl_vector_pipeline_value<Eigen::Vector3f>;
		if (!prev_n_ptr) prev_n_ptr=std::make_unique<n_type>(q_);
		auto && prev_n=dynamic_cast<n_type &>(*prev_n_ptr);
		auto && prev_n_vec=prev_n.vector();

		opencl_vector_pipeline_value_extractor<Eigen::Vector3f> v_e(q_);
		auto && vertices=v_e(v);
		opencl_vector_pipeline_value_extractor<Eigen::Vector3f> n_e(q_);
		auto && normals=n_e(n);

		prev_v_vec.resize(vertices.size(),q_);
		prev_n_vec.resize(normals.size(),q_);

		boost::compute::copy(vertices.begin(),vertices.end(),prev_v_vec.begin(),q_);
		boost::compute::copy(normals.begin(),normals.end(),prev_n_vec.begin(),q_);

		return std::make_tuple(std::move(prev_v_ptr),std::move(prev_n_ptr));

	}


}
