#include <seng499/msrc_file_system_depth_device.hpp>


#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <catch.hpp>


SCENARIO("msrc_file_system_depth_device loads depth information from MSRC 7Scene files in the file system", "[seng499][depth_device][file_system_depth_device][msrc_file_system_depth_device]") {
	
	GIVEN("A valid msrc_file_system_depth_device") {
		
		boost::filesystem::path test_data_path("data/test/msrc_file_system_depth_device");
		seng499::msrc_file_system_depth_device_frame_factory fac;
		seng499::msrc_file_system_depth_device_filter filt;
		seng499::file_system_depth_device fsdd(std::move(test_data_path),fac, &filt);
		

		WHEN("Returning the width and height") {

			auto w = fsdd.width();
			auto h = fsdd.height();

			THEN("A width of 480 is returned") {

				CHECK(w==480U);

			}

			THEN("A height of 640 is returned") {

				CHECK(h==640U);

			}

		}

		WHEN("It is invoked on the test image data/test/msrc_file_system_depth_device/frame-000000.depth.png") {

			auto frame=fsdd();

			THEN("A non-zero number of elements are returned") {

				CHECK(frame.size() != 0);
	
			}

		}

		
	
	}
	
	
	GIVEN("An invalid msrc_file_system_depth_device") {

		
		boost::filesystem::path test_data_path("data/test/msrc_file_system_depth_device/garbage");
		seng499::msrc_file_system_depth_device_frame_factory fac;
		seng499::file_system_depth_device fsdd(std::move(test_data_path),fac);

		WHEN("It is invoked on the test image data/test/msrc_file_system_depth_device/00-file_to_small.png") {

			CHECK_THROWS_AS(fsdd(), std::runtime_error);
			
		}


	}

}
