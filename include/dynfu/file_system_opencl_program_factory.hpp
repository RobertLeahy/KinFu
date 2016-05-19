/**
 *	\file
 */


#pragma once


#include <boost/compute/context.hpp>
#include <boost/compute/program.hpp>
#include <boost/filesystem.hpp>
#include <dynfu/opencl_program_factory.hpp>
#include <string>


namespace dynfu {
	
	
	/**
	 *
	 */
	class file_system_opencl_program_factory : public opencl_program_factory {
		
		
		private:
		
		
			boost::filesystem::path root_;
			boost::compute::context ctx_;
			
			
		public:
		
		
			file_system_opencl_program_factory () = delete;
			
			
			/**
			 *	Creates a new file_system_opencl_program_factory which
			 *	loads programs from the file system based on name
			 *	relative to some path.
			 *
			 *	\param [in] path
			 *		The path which will be searched for OpenCL source
			 *		files.
			 *	\param [in] ctx
			 *		The boost::compute::context which shall be used to
			 *		build programs.
			 */
			file_system_opencl_program_factory (boost::filesystem::path path, boost::compute::context ctx);
			
			
			virtual boost::compute::program operator () (const std::string &) override; 
		
		
	};
	
	
}
