/**
 *	\file
 */


#pragma once


#include <boost/compute/context.hpp>
#include <boost/compute/program.hpp>
#include <kinfu/filesystem.hpp>
#include <kinfu/opencl_program_factory.hpp>
#include <string>


namespace kinfu {
	
	
	/**
	 *	An \ref opencl_program_factory which loads and
	 *	compiles OpenCL files from the file system.
	 */
	class file_system_opencl_program_factory : public opencl_program_factory {
		
		
		private:
		
		
			filesystem::path root_;
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
			file_system_opencl_program_factory (filesystem::path path, boost::compute::context ctx);
			
			
			virtual boost::compute::program operator () (const std::string &) override;
		
		
	};
	
	
}
