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

		dynfu::measurement_pipeline_block::value_type vn;
		using std::swap;
		swap(vn,vn_);
		timer t;
		vn_=(*mpb_)(*frame_,width_,height_,k_,std::move(vn));
		mpbt_=t.elapsed();
		
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
	
	
	void kinect_fusion::operator () () {
		
		get_frame();
		get_vertex_and_normal_maps();
		
		//	TODO
		
	}
	
	
	kinect_fusion::timer::duration kinect_fusion::depth_device_elapsed () const {
		
		return ddt_;
		
	}
	
	
	kinect_fusion::timer::duration kinect_fusion::measurement_pipeline_block_elapsed () const {
		
		return mpbt_;
		
	}
	
	
}
