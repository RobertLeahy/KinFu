/**
 *	\file
 */


#pragma once


#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/vector.hpp>
#include <seng499/opencl_pipeline_value.hpp>
#include <seng499/pipeline_value.hpp>
#include <seng499/optional.hpp>
#include <iterator>
#include <utility>
#include <vector>


namespace seng499 {
	
	
	/**
	 *	Represents a vector of values stored on the GPU.
	 *
	 *	If \em T is not trivial the behaviour is undefined.
	 *
	 *	\tparam T
	 *		The element type.
	 */
	template <typename T>
	class opencl_vector_pipeline_value final : public opencl_pipeline_value<std::vector<T>> {
		
		
		private:
		
		
			using base=opencl_pipeline_value<std::vector<T>>;
			
			
			boost::compute::vector<T> gpu_;
			std::vector<T> cpu_;
			bool dirty_;
			
			
			void download () {
				
				if (!dirty_) return;
				
				cpu_.clear();
				//	Must be a mutable lvalue reference
				auto q=base::command_queue();
				boost::compute::copy(gpu_.begin(),gpu_.end(),std::back_inserter(cpu_),q);
				
				dirty_=false;
				
			}
			
			
		public:
		
		
			/**
			 *	Creates a new opencl_vector_pipeline_value.
			 *
			 *	\param [in] q
			 *		The boost::compute::command_queue which will
			 *		be associated with this pipeline value.
			 */
			explicit opencl_vector_pipeline_value (boost::compute::command_queue q) : base(q), gpu_(q.get_context()), dirty_(false) {	}
			
			
			/**
			 *	Returns a reference to the wrapped boost::compute::vector.
			 *
			 *	Invoking this method causes the CPU copy (if any) of the
			 *	GPU data to be considered dirty (i.e. it will be downloaded
			 *	the next time it is retrieved).
			 *
			 *	Performing operations on the referenced object using any
			 *	boost::compute::command_queue except for one which compares
			 *	equal to the boost::compute::command_queue associated with
			 *	this object causes undefined behaviour.
			 *
			 *	\return
			 *		A reference to the wrapped GPU storage.
			 */
			boost::compute::vector<T> & vector () noexcept {
				
				dirty_=true;
				return gpu_;
				
			}
			/**
			 *	Returns a reference to the wrapped boost::compute::vector.
			 *
			 *	\return
			 *		A reference to the wrapped GPU storage.
			 */
			const boost::compute::vector<T> & vector () const noexcept {
				
				return gpu_;
				
			}
			
			
			virtual const std::vector<T> & get () override {
				
				download();
				return cpu_;
				
			}
		
		
	};
	
	
	/**
	 *	Allows vector pipeline values to be consumed on the
	 *	GPU whether the value-in-question lives on the CPU
	 *	or the GPU.  GPU values will be seamlessly reused
	 *	on the GPU without copying, uploading, or downloading
	 *	when possible.  CPU values shall be uploaded.  GPU
	 *	values which cannot be seamlessly reused on the GPU
	 *	due to a difference in device or context shall be
	 *	downloaded and then uploaded.
	 *
	 *	\sa pipeline_value, opencl_vector_pipeline_value, opencl_pipeline_value
	 *
	 *	\tparam T
	 *		The element type.
	 */
	template <typename T>
	class opencl_vector_pipeline_value_extractor {
		
		
		private:
		
		
			boost::compute::command_queue q_;
			optional<boost::compute::vector<T>> v_;
			
			
			boost::compute::vector<T> & from_vector (const std::vector<T> & v) {
				
				if (!v_) v_.emplace(q_.get_context());
				v_->assign(v.begin(),v.end(),q_);
				return *v_;
				
			}
			
			
		public:
		
		
			opencl_vector_pipeline_value_extractor () = delete;
			opencl_vector_pipeline_value_extractor (const opencl_vector_pipeline_value_extractor &) = delete;
			opencl_vector_pipeline_value_extractor (opencl_vector_pipeline_value_extractor &&) = delete;
			opencl_vector_pipeline_value_extractor & operator = (const opencl_vector_pipeline_value_extractor &) = delete;
			opencl_vector_pipeline_value_extractor & operator = (opencl_vector_pipeline_value_extractor &&) = delete;
			
			
			/**
			 *	Creates an opencl_vector_pipeline_value_extractor.
			 *
			 *	\param [in] q
			 *		The boost::compute::command_queue which shall be
			 *		used to manage the boost::compute::vector which
			 *		holds a copy of the pipeline value (if necessary)
			 *		and whose context and device shall be used for
			 *		compatibility checks.
			 */
			explicit opencl_vector_pipeline_value_extractor (boost::compute::command_queue q) : q_(std::move(q)) {	}
			
			
			/**
			 *	Obtains a boost::compute::vector in the context and
			 *	on the device associated with the boost::compute::command_queue
			 *	associated with this object which contains a copy of
			 *	the data in a particular \ref pipeline_value.
			 *
			 *	\param [in] pv
			 *		The \ref pipeline_value whose value shall be extracted.
			 *
			 *	\return
			 *		A reference to a boost::compute::vector.  This reference
			 *		shall remain valid as long as the lifetime of both this
			 *		object and \em pv persist.
			 */
			boost::compute::vector<T> & operator () (pipeline_value<std::vector<T>> & pv) {
				
				auto ptr=dynamic_cast<opencl_vector_pipeline_value<T> *>(&pv);
				if (!ptr) return from_vector(pv.get());
				
				using compatibility=typename opencl_pipeline_value<std::vector<T>>::compatibility;
				switch (ptr->check(q_)) {
					
					case compatibility::same_context:
						ptr->command_queue().finish();
					case compatibility::same_queue:
						return ptr->vector();
					default:
						break;
					
				}
				
				return from_vector(ptr->get());
				
			}
		
		
	};
	
	
}
