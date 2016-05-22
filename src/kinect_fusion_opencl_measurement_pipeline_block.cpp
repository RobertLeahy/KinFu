#include <boost/compute/buffer.hpp>
#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <tuple>


//	This is evil black magic
namespace boost {
	
	
	namespace compute {
		
		
		namespace detail {
			
			
			template <>
			struct type_name_trait<Eigen::Vector3f> {
				
				
				static const char * value () noexcept {
					
					return "Eigen::Vector3f";
					
				}
				
				
			};
			
			
		}
		
		
	}
	
	
}


namespace dynfu {
	
	
	static boost::compute::kernel get_bilateral_kernel (opencl_program_factory & opf) {
		
		auto p=opf("bilateral");
		return boost::compute::kernel(p,"bilateral_filter");
		
	}
	
	
	static boost::compute::kernel get_v_kernel (opencl_program_factory & opf) {
		
		auto p=opf("vertex_map");
		return boost::compute::kernel(p,"vertex_map");
		
	}
	
	
	static boost::compute::kernel get_n_kernel (opencl_program_factory & opf) {
		
		auto p=opf("normal_map");
		return boost::compute::kernel(p,"normal_map");
		
	}
	
	
	kinect_fusion_opencl_measurement_pipeline_block::kinect_fusion_opencl_measurement_pipeline_block (boost::compute::command_queue q, opencl_program_factory & opf)
		:	ve_(std::move(q)),
			bilateral_kernel_(get_bilateral_kernel(opf)),
			v_kernel_(get_v_kernel(opf)),
			n_kernel_(get_n_kernel(opf)),
			v_(ve_.command_queue().get_context()),
			sigma_s_(2.0f),
			sigma_r_(1.0f)
	{	}
	
	
	kinect_fusion_opencl_measurement_pipeline_block::value_type kinect_fusion_opencl_measurement_pipeline_block::operator () (
		depth_device::value_type::element_type & frame,
		std::size_t width,
		std::size_t height,
		Eigen::Matrix3f k,
		kinect_fusion_opencl_measurement_pipeline_block::value_type v
	) {
		
		//	Bilateral filtering
		//
		//	Kernel arguments are:
		//
		//	0:	Source
		//	1:	Destination
		//	2:	sigma_s^-2
		//	3:	sigma_r^-2
		//	4:	Width
		//	5:	Height
		auto && vec=ve_(frame);
		auto s=vec.size();
		auto q=ve_.command_queue();
		v_.resize(s,q);
		bilateral_kernel_.set_arg(0,vec);
		bilateral_kernel_.set_arg(1,v_);
		bilateral_kernel_.set_arg(2,1.0f/(sigma_s_*sigma_s_));
		bilateral_kernel_.set_arg(3,1.0f/(sigma_r_*sigma_r_));
		bilateral_kernel_.set_arg(4,std::uint32_t(width));
		bilateral_kernel_.set_arg(5,std::uint32_t(height));
		std::size_t extent []={width,height};
		q.enqueue_nd_range_kernel(bilateral_kernel_,2,nullptr,extent,nullptr);
		
		auto && vertices_ptr=std::get<0>(v);
		auto && normals_ptr=std::get<1>(v);
		using type=opencl_vector_pipeline_value<Eigen::Vector3f>;
		if (!vertices_ptr) vertices_ptr=std::make_unique<type>(q);
		if (!normals_ptr) normals_ptr=std::make_unique<type>(q);
		auto && vertices=dynamic_cast<type &>(*vertices_ptr).vector();
		auto && normals=dynamic_cast<type &>(*normals_ptr).vector();
		vertices.resize(s,q);
		normals.resize(s,q);
		
		//	Vertex map
		//
		//	Kernel arguments are:
		//
		//	0:	Source (bilaterally filtered image)
		//	1:	Destination
		//	2:	K^-1
		//	3:	Width
		//	4:	Height
		Eigen::Matrix3f k_inv=k.inverse();
		k_inv.transposeInPlace();
		v_kernel_.set_arg(0,v_);
		v_kernel_.set_arg(1,vertices);
		boost::compute::buffer k_inv_buf(q.get_context(),sizeof(k_inv),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,k_inv.data());
		v_kernel_.set_arg(2,k_inv_buf);
		v_kernel_.set_arg(3,std::uint32_t(width));
		v_kernel_.set_arg(4,std::uint32_t(height));
		q.enqueue_nd_range_kernel(v_kernel_,2,nullptr,extent,nullptr);
		
		//	Normal map
		//
		//	Kernel arguments are:
		//
		//	0:	Source (vertex map)
		//	1:	Destination
		//	2:	Width
		//	3:	Height
		n_kernel_.set_arg(0,vertices);
		n_kernel_.set_arg(1,normals);
		n_kernel_.set_arg(2,std::uint32_t(width));
		n_kernel_.set_arg(3,std::uint32_t(height));
		q.enqueue_nd_range_kernel(n_kernel_,2,nullptr,extent,nullptr);
		
		return v;
		
	}
	

}