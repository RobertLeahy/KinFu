/**
 *	\file
 */


#pragma once


#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/vector.hpp>
#include <seng499/opencl_pipeline_value.hpp>
#include <iterator>
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
	
	
}
