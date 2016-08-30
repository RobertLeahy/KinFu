/**
 *  \file
 */


#pragma once


#include <boost/compute/context.hpp>
#include <boost/compute/exception/opencl_error.hpp>
#include <boost/compute/program.hpp>
#include <kinfu/filesystem.hpp>
#include <string>


namespace kinfu {
	
	
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
			
			
		protected:
		
		
			opencl_build_error (const boost::compute::program &, const boost::compute::opencl_error &, std::string);
			
			
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
	
	
	/**
	 *	A subclass of \ref opencl_build_error which represents an error
	 *	encountered building an OpenCL program whose source code was obtained
	 *	from a file in the file system.  This allows the file to be included
	 *	in the exception's message which eases the debugging process.
	 */
	class opencl_file_build_error : public opencl_build_error {
		
		
		private:
		
		
			filesystem::path p_;
			
			
			opencl_file_build_error (const boost::compute::program &, const boost::compute::opencl_error &, const filesystem::path &);
			
			
		public:
		
		
			opencl_file_build_error () = delete;
			opencl_file_build_error (const opencl_file_build_error &) = default;
			opencl_file_build_error (opencl_file_build_error &&) = default;
			opencl_file_build_error & operator = (const opencl_file_build_error &) = default;
			opencl_file_build_error & operator = (opencl_file_build_error &&) = default;
			
			
			/**
			 *	Throws an instance of this class or \em err as appropriate.
			 *
			 *	If \em err does not indicate that there was a problem with the build
			 *	then \em err shall be rethrown, otherwise an instance of this class
			 *	created from \em p, \em err, and \em path shall be thrown.
			 *
			 *	\param [in] p
			 *		The boost::compute::program which was being built when \em err
			 *		was thrown.
			 *	\param [in] err
			 *		The boost::compute::opencl_error which was thrown when attempting
			 *		to build \em p.
			 *	\param [in] path
			 *		The path on the file system from which the source for \em p was
			 *		drawn.
			 */
			[[noreturn]]
			static void raise (const boost::compute::program & p, const boost::compute::opencl_error & err, const filesystem::path & path);
			
			
			/**
			 *	Builds an OpenCL program from source held in a file on the file
			 *	system, throwing an instance of this class if the build fails and
			 *	the error is appropriate to be represented by this class.
			 *
			 *	\param [in] path
			 *		The path on the file system from which the source for the program
			 *		to be built shall be drawn.
			 *	\param [in] ctx
			 *		The OpenCL context within which to build the program.
			 *
			 *	\return
			 *		The program built from the source held in file \em path.
			 */
			static boost::compute::program build (const filesystem::path & path, const boost::compute::context & ctx);
			
			
			/**
			 *	Retrieves an object which represents the location in the file system
			 *	from which the source for the program being built was drawn.
			 *
			 *	\return
			 *		A path object.
			 */
			const filesystem::path & path () const noexcept;
		
		
	};
	
	
}
