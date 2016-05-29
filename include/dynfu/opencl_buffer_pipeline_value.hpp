/**
 *	\file
 */


#pragma once


#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <dynfu/opencl_pipeline_value.hpp>
#include <dynfu/optional.hpp>
#include <utility>


namespace dynfu {
	
	
	/**
	 *	Represents a buffer which stores a single value
	 *	on the GPU.
	 *
	 *	\tparam T
	 *		The element type.  Objects of this type must
	 *		be able to be copied and restored by copying
	 *		their raw bytes.  Objects of this type must use
	 *		the same representation in OpenCL as they use
	 *		on the CPU.
	 */
	template <typename T>
	class opencl_buffer_pipeline_value : public opencl_pipeline_value<T> {
		
		
		private:
		
		
			using base=opencl_pipeline_value<T>;
			
			
			boost::compute::buffer b_;
			optional<T> cpu_;
			
			
			void download () {
				
				if (cpu_) return;
				
				cpu_.emplace();
				base::command_queue().enqueue_read_buffer(b_,0,sizeof(T),&(*cpu_));
				
			}
			
			
		public:
		
		
			/**
			 *	Creates a new opencl_buffer_pipeline_value by
			 *	allocating space in OpenCL.
			 *
			 *	\param [in] q
			 *		The boost::compute::command_queue which will
			 *		be associated with this pipeline value.
			 */
			opencl_buffer_pipeline_value (boost::compute::command_queue q)
				:	base(q),
					b_(q.get_context(),sizeof(T))
			{	}
			
			
			/**
			 *	Returns a reference to the wrapped boost::compute::buffer.
			 *
			 *	Invoking this method causes the CPU copy (if any) of the
			 *	OpenCL data to be considered dirty.  This will cause it to
			 *	be redownloaded the next time it is retrieved.
			 *
			 *	\return
			 *		A reference to the wrapped OpenCL storage.
			 */
			boost::compute::buffer & buffer () noexcept {
				
				cpu_=nullopt;
				return b_;
				
			}
			/**
			 *	Returns a reference to the wwrapped boost::compute::buffer.
			 *
			 *	\return
			 *		A reference to the wrapped OpenCL storage.
			 */
			const boost::compute::buffer & buffer () const noexcept {
				
				return b_;
				
			}
			
			
			virtual const T & get () override {
				
				download();
				return *cpu_;
				
			}
		
		
	};
	
	
	/**
	 *	Allows pipeline values stored in OpenCL using
	 *	buffers to be consumed on the GPU whether the
	 *	value-in-question lives on the CPU or in OpenCL.
	 *	OpenCL values will be seamlessly used without
	 *	copying if possible, CPU values shall be uploaded.
	 *
	 *	\sa pipeline_value, opencl_pipeline_value, opencl_buffer_pipeline_value
	 *
	 *	\tparam T
	 *		The element type.
	 */
	template <typename T>
	class opencl_buffer_pipeline_value_extractor {
		
		
		private:
		
		
			boost::compute::command_queue q_;
			optional<boost::compute::buffer> b_;
			
			
			boost::compute::buffer & from_value (const T & v) {
				
				if (!b_) b_.emplace(q_.get_context(),sizeof(T));
				q_.enqueue_write_buffer(*b_,0,sizeof(T),&v);
				return *b_;
				
			}
			
			
		public:
		
		
			opencl_buffer_pipeline_value_extractor () = delete;
			opencl_buffer_pipeline_value_extractor (const opencl_buffer_pipeline_value_extractor &) = delete;
			opencl_buffer_pipeline_value_extractor (opencl_buffer_pipeline_value_extractor &&) = delete;
			opencl_buffer_pipeline_value_extractor & operator = (const opencl_buffer_pipeline_value_extractor &) = delete;
			opencl_buffer_pipeline_value_extractor & operator = (opencl_buffer_pipeline_value_extractor &&) = delete;
			
			
			/**
			 *	Creates an opencl_buffer_pipeline_value_extractor.
			 *
			 *	\param [in] q
			 *		The boost::compute::command_queue which shall be
			 *		used to manage the boost::compute::buffer which
			 *		holds a copy of the pipeline value (if necessary)
			 *		and whose context and device shall be used for
			 *		compatibility checks.
			 */
			explicit opencl_buffer_pipeline_value_extractor (boost::compute::command_queue q) : q_(std::move(q)) {	}
			
			
			boost::compute::buffer & operator () (pipeline_value<T> & pv) {
				
				auto ptr=dynamic_cast<opencl_buffer_pipeline_value<T> *>(&pv);
				if (!ptr) return from_value(pv.get());
				
				using compatibility=typename opencl_pipeline_value<T>::compatibility;
				switch (ptr->check(q_)) {
					
					case compatibility::same_context:
						ptr->command_queue().finish();
					case compatibility::same_queue:
						return ptr->buffer();
					default:
						break;
					
				}
				
				return from_value(ptr->get());
				
			}
			
			
			/**
			 *	Retrieves the boost::compute::command_queue
			 *	associated with this object.
			 *
			 *	\return
			 *		A boost::compute::command_queue.
			 */
			boost::compute::command_queue command_queue () const {
				
				return q_;
				
			}
		
		
	};
	
	
}
