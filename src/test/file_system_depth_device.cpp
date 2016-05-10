#include <seng499/file_system_depth_device.hpp>

#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <experimental/optional>
#include <catch.hpp>


SCENARIO("file_system_depth_device is used to debounce the rate at which frames can be acquired", "[seng499][depth_device][file_system_depth_device]") {
	
	GIVEN("30 fps") {
		
		// A fake implementatio of a file_system_depth_device for testing debouncing
		class fake_file_system_depth_device : public seng499::file_system_depth_device::file_system_depth_device {

			public:
			
				using file_system_depth_device::file_system_depth_device;
				
				std::size_t width() const noexcept { return 0; }
				std::size_t height() const noexcept { return 0; }
				Eigen::Matrix3f k() const noexcept { return Eigen::Matrix3f::Zero(); }

				std::vector<float> get_file_system_frame(std::vector<float> vec) const {
					return vec;
				}

		};
		
		std::vector<float> frame;
		int fps = 30;
		boost::filesystem::path fake_path("src/test/fakepath");	
		
		fake_file_system_depth_device fsdd(fps, fake_path);
		
		WHEN("The () operator is invoked in succession") {
			
			fsdd(frame);
			using clock=std::chrono::high_resolution_clock;
			auto start=clock::now();
			fsdd(frame);
			auto delta=clock::now()-start;
			
			THEN("At least 1/30 seconds occurs before the second invocation returns") {
				
				auto delta_ms=std::chrono::duration_cast<std::chrono::milliseconds>(delta);
				CHECK(delta_ms>=std::chrono::milliseconds(33));
			
			}
			
			
		}
	
	}
	
	
}