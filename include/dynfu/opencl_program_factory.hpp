/**
 *	\file
 */


#pragma once


#include <boost/compute/program.hpp>
#include <string>


namespace dynfu {
	
	
	/**
	 *	An abstract base class which allows derived classes
	 *	to provide OpenCL programs to consumers thereof.
	 */
	class opencl_program_factory {
		
		
		public:
		
		
			opencl_program_factory () = default;
			opencl_program_factory (const opencl_program_factory &) = delete;
			opencl_program_factory (opencl_program_factory &&) = delete;
			opencl_program_factory & operator = (const opencl_program_factory &) = delete;
			opencl_program_factory & operator = (opencl_program_factory &&) = delete;
			
			
			/**
			 *	Allows derived classes to be cleaned up
			 *	through pointer or reference to base.
			 */
			virtual ~opencl_program_factory () noexcept;
			
			
			/**
			 *	Retrieves a boost::compute::program given
			 *	a program name.
			 *
			 *	\param [in] name
			 *		The name of the program to retrieve.
			 *
			 *	\return
			 *		The program referred to by \em name.
			 */
			virtual boost::compute::program operator () (const std::string & name) = 0;
		
		
	};
	
	
}
