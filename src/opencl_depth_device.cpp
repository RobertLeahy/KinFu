#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/async/future.hpp>
#include <kinfu/opencl_depth_device.hpp>
#include <kinfu/opencl_vector_pipeline_value.hpp>
#include <memory>
#include <utility>


namespace kinfu {
	
	
	opencl_depth_device::opencl_depth_device (depth_device & dev, boost::compute::command_queue q)
		:	depth_device_decorator(dev),
			q_(std::move(q))
	{	}
	
	
	opencl_depth_device::value_type opencl_depth_device::operator () (value_type v) {
		
		using base_type=opencl_vector_pipeline_value<buffer_type::value_type>;
		class pipeline_value final : public base_type {
			
			
			public:
			
			
				value_type inner;
				boost::compute::future<boost::compute::vector<buffer_type::value_type>::iterator> future;
				
				
				using base_type::base_type;
				
				
				~pipeline_value () noexcept {
					
					//	Because OpenCL may be still using
					//	things until the asynchronous task
					//	completes
					if (future.valid()) future.wait();
					
				}
				
				
				virtual const buffer_type & get () override {
					
					return inner->get();
					
				}
			
			
		};
		
		bool was(v);
		if (!was) v=std::make_unique<pipeline_value>(q_);
		auto && pv=static_cast<pipeline_value &>(*v.get());
		//	We don't want undefined behaviour/race
		//	conditions
		if (was) pv.future.wait();
		
		value_type pvi;
		using std::swap;
		swap(pvi,pv.inner);
		pv.inner=dev_(std::move(pvi));
		auto && vi=pv.inner->get();
		
		auto && vec=pv.vector();
		vec.resize(vi.size(),q_);
		pv.future=boost::compute::copy_async(vi.begin(),vi.end(),vec.begin(),q_);
		
		return v;
		
	}
	
	
}
