#include <dynfu/kinect_fusion.hpp>
#include <Eigen/Dense>
#include <tuple>
#include <utility>


namespace dynfu {
	
	
	void kinect_fusion::get_frame () {
		
		dynfu::depth_device::value_type f;
		using std::swap;
		swap(f,frame_);
		timer t;
		frame_=(*dd_)(std::move(f));
		ddt_=t.elapsed();
		
	}
	
	
	void kinect_fusion::get_vertex_and_normal_maps () {
		
		dynfu::measurement_pipeline_block::vertex_value_type v;
		dynfu::measurement_pipeline_block::normal_value_type n;
		using std::swap;
		swap(v,v_);
		swap(n,n_);
		timer t;
		std::tie(v_,n_)=(*mpb_)(*frame_,width_,height_,k_,std::make_tuple(std::move(v),std::move(n)));
		mpbt_=t.elapsed();
		
	}


	void kinect_fusion::get_tgk () {

		dynfu::pose_estimation_pipeline_block::value_type t_g_k;
		using std::swap;
		swap(t_g_k,t_g_k_);
		timer t;
		t_g_k_=(*pepb_)(*v_,*n_,prev_v_.get(),prev_n_.get(),k_,std::move(t_g_k));
		pepbt_=t.elapsed();

	}
	
	
	kinect_fusion::kinect_fusion () noexcept
		:	dd_(nullptr),
			k_(Eigen::Matrix3f::Zero()),
			width_(0),
			height_(0),
			mpb_(nullptr),
			pepb_(nullptr),
			sppb_(nullptr),
			urpb_(nullptr)
	{	}
	
	
	void kinect_fusion::depth_device (dynfu::depth_device & dd) noexcept {
		
		dd_=&dd;
		k_=dd.k();
		width_=dd.width();
		height_=dd.height();
		
	}
	
	
	void kinect_fusion::measurement_pipeline_block (dynfu::measurement_pipeline_block & mpb) noexcept {
		
		mpb_=&mpb;
		
	}


	void kinect_fusion::pose_estimation_pipeline_block (dynfu::pose_estimation_pipeline_block & pepb) noexcept {

		pepb_=&pepb;

	}
	
	
	void kinect_fusion::operator () () {
		
		get_frame();
		get_vertex_and_normal_maps();
		get_tgk();
		
		//	TODO
		
	}
	
	
	kinect_fusion::timer::duration kinect_fusion::depth_device_elapsed () const {
		
		return ddt_;
		
	}
	
	
	kinect_fusion::timer::duration kinect_fusion::measurement_pipeline_block_elapsed () const {
		
		return mpbt_;
		
	}


	kinect_fusion::timer::duration kinect_fusion::pose_estimation_pipeline_block_elapsed () const {

		return pepbt_;

	}
	
	
}
