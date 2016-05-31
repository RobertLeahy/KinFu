/**
 *  \file
 */


#pragma once


#include <boost/compute/exception/opencl_error.hpp>
#include <boost/compute/program.hpp>
#include <string>


namespace dynfu {
	
	
	/**
	 *	A subclass of boost::compute::opencl_error which specifically
	 *	represents an error encountered while attempting to build
	 *	an OpenCL program.
	 */
	class opencl_build_error : public boost::compute::opencl_error {
		
		
		private:
		
		
			boost::compute::program p_;
			std::string w_;
			
			
			opencl_build_error (const boost::compute::program &, const boost::compute::opencl_error &);
			
			
		public:
		
		
			opencl_build_error () = delete;
			opencl_build_error (const opencl_build_error &) = default;
			opencl_build_error (opencl_build_error &&) = default;
			opencl_build_error & operator = (const opencl_build_error &) = default;
			opencl_build_error & operator = (opencl_build_error &) = default;
			
			
			/**
			 *	Throws an instance of this class or \em err as appropriate.
			 *
			 *	If \em err does not indicate that there was a problem with the build
			 *	then \em err shall be rethrown, otherwise an instance of this class
			 *	created from \em p and \em err shall be thrown.
			 *
			 *	\param [in] p
			 *		The boost::compute::program which was being built when \em err
			 *		was thrown.
			 *	\param [in] err
			 *		The boost::compute::opencl_error which was thrown when attempting
			 *		to build \em p.
			 */
			[[noreturn]]
			static void raise (const boost::compute::program & p, const boost::compute::opencl_error & err);
			
			
			/**
			 *	Builds an OpenCL program throwing an instance of this class if the
			 *	build fails and the error is appropriate to be represented by this
			 *	class.
			 *
			 *	\param [in] p
			 *		The program to be built.
			 */
			static void build (boost::compute::program & p);
			
			
			/**
			 *	Retrieves the boost::compute::program which was being built
			 *	when the error which caused this exception to be thrown occurred.
			 *
			 *	\return
			 *		The program.
			 */
			boost::compute::program program () const;
			
			
			virtual const char * what () const noexcept override;
		
		
	};
	
	
}
