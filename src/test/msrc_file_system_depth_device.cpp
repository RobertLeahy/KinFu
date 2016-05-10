#include <seng499/msrc_file_system_depth_device.hpp>


#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <catch.hpp>


SCENARIO("msrc_file_system_depth_device loads depth information from MSRC 7Scene files in the file system", "[seng499][depth_device][file_system_depth_device][msrc_file_system_depth_device]") {
	
	GIVEN("An msrc_file_system_depth_device") {
		
		boost::filesystem::path test_data_path("data/test/msrc_file_system_depth_device");
		seng499::msrc_file_system_depth_device_frame_factory fac;
		seng499::msrc_file_system_depth_device_filter filt;
		seng499::file_system_depth_device fsdd(std::move(test_data_path),fac, &filt);
		

		WHEN("Returning the width and height") {

			std::size_t w = fsdd.width();
			std::size_t h = fsdd.height();

			THEN("A width of 640 is returned") {

				CHECK(w==640U);

			}

			THEN("A height of 480 is returned") {

				CHECK(h==480U);

			}

		}

		WHEN("It is invoked on the test image data/test/msrc_file_system_depth_device/frame-000000.depth.png") {

			std::vector<float> frame=fsdd();

			THEN("A nonzero number elements are returned") {

				CHECK(frame.size() != 0);
	
			}

		}

		
	
	}
	
	
}
