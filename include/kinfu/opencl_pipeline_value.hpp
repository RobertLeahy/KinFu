/**
 *	\file
 */


#pragma once


#include <boost/compute/command_queue.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <kinfu/pipeline_value.hpp>
#include <utility>


namespace kinfu {
	
	
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
		
		
			/**
			 *	An enumeration of the different types of compatibility
			 *	between two opencl_pipeline_value objects.
			 */
			enum class compatibility {
				
				same_queue,	/**<	This object and the other object have the same queue, context, and device.	*/
				same_context,	/**<	This object and the other object have the same context, and device.	*/
				same_device,	/**<	This object and the other object have the same device.	*/
				none	/**<	This object and the other object have nothing in common.	*/
				
			};
			
			
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
			
			
			/**
			 *	Checks the compatibility of this object with another
			 *	OpenCL command queue, context, and device.
			 *
			 *	\param [in] q
			 *		The command queue to check against.
			 *
			 *	\return
			 *		An enumeration value indicating the level of
			 *		compatibility between this object and \em q.
			 */
			compatibility check (const boost::compute::command_queue & q) const noexcept {
				
				if (q_.get_device()!=q.get_device()) return compatibility::none;
				if (q_.get_context()!=q.get_context()) return compatibility::same_device;
				if (q_!=q) return compatibility::same_context;
				return compatibility::same_queue;
				
			}
		
		
	};
	
	
}
