/**
 *	\file
 */


#pragma once


#include <boost/compute/command_queue.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <seng499/pipeline_value.hpp>
#include <utility>


namespace seng499 {
	
	
	/**
	 *	An abstract base class which refines the concept of
	 *	a pipeline value by associating it with an OpenCL
	 *	command queue.
	 *
	 *	\tparam T
	 *		The type of value to represent.
	 */
	template <typename T>
	class opencl_pipeline_value : public pipeline_value<T> {
		
		
		private:
		
		
			boost::compute::command_queue q_;
			
			
		public:
		
		
			opencl_pipeline_value () = delete;
			
			
			/**
			 *	Creates a new opencl_pipeline_value.
			 *
			 *	\param [in] q
			 *		The boost::compute::command_queue which will
			 *		be associated with this pipeline value.
			 */
			explicit opencl_pipeline_value (boost::compute::command_queue q) : q_(std::move(q)) {	}
			
			
			/**
			 *	Retrieves the boost::compute::command_queue
			 *	associated with this object.
			 *
			 *	\return
			 *		The command queue.
			 */
			boost::compute::command_queue command_queue () const {
				
				return q_;
				
			}
			
			
			/**
			 *	Retrieves the boost::compute::context associated
			 *	with this object.
			 *
			 *	\return
			 *		The context.
			 */
			boost::compute::context context () const {
				
				return q_.get_context();
				
			}
			
			
			/**
			 *	Retrieves the boost::compute::device associated
			 *	with this object.
			 *
			 *	\return
			 *		The device.
			 */
			boost::compute::device device () const {
				
				return q_.get_device();
				
			}
		
		
	};
	
	
}
