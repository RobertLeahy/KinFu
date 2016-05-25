#include <boost/compute.hpp>
#include <dynfu/file_system_depth_device.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/kinect_fusion.hpp>
#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>
#include <dynfu/msrc_file_system_depth_device.hpp>
#include <dynfu/opencl_depth_device.hpp>
#include <dynfu/path.hpp>
#include <dynfu/timer.hpp>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>


void main_impl (int, char **) {
	
	dynfu::filesystem::path pp(dynfu::current_executable_parent_path());
	
	auto d=boost::compute::system::default_device();
	boost::compute::context ctx(d);
	boost::compute::command_queue q(ctx,d);
	
	dynfu::msrc_file_system_depth_device_frame_factory ff;
	dynfu::msrc_file_system_depth_device_filter f;
	dynfu::file_system_depth_device ddi(pp/".."/"data/test/msrc_file_system_depth_device",ff,&f);
	dynfu::opencl_depth_device dd(ddi,q);
	
	dynfu::file_system_opencl_program_factory opf(pp/".."/"cl",ctx);
	dynfu::kinect_fusion_opencl_measurement_pipeline_block mpb(q,opf,20,2.0f,1.0f);
	
	dynfu::kinect_fusion kf;
	kf.depth_device(dd);
	kf.measurement_pipeline_block(mpb);

	std::size_t total(0);
	std::size_t frames(0);
	while (ddi) {
			
		dynfu::timer t;
		kf();
		//	We do this because we're not actually consuming
		//	the generated values, in the future this shouldn't
		//	be necessary as the pipeline_value objects we consume
		//	will do this or equivalent internally as they'll have
		//	to wait for their offered values to become available
		//	on the GPU before they download them
		q.finish();
		auto e=t.elapsed_ms();
		std::cout << "Depth frame took " << std::chrono::duration_cast<std::chrono::milliseconds>(kf.depth_device_elapsed()).count() << "ms" << std::endl;
		std::cout << "Measurement took " << std::chrono::duration_cast<std::chrono::milliseconds>(kf.measurement_pipeline_block_elapsed()).count() << "ms" << std::endl;
		std::cout << "Frame took " << e.count() << "ms" << std::endl;
		total+=e.count();
		++frames;
		
	}
	
	std::cout << "Frames: " << frames << std::endl;
	std::size_t avg=(frames==0) ? 0 : (total/frames);
	std::cout << "Average time per frame: " << avg << "ms" << std::endl;
	
}


int main (int argc, char ** argv) {
	
	try {
		
		try {
			
			main_impl(argc,argv);
			
		} catch (const std::exception & ex) {
			
			std::cerr << "ERROR: " << ex.what() << std::endl;
			throw;
			
		} catch (...) {
			
			std::cerr << "ERROR" << std::endl;
			throw;
			
		}
		
	} catch (...) {
		
		return EXIT_FAILURE;
		
	}
	
}