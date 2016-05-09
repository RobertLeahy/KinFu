#include <seng499/file_system_depth_device.hpp>

#include <Eigen/Dense>
#include <ctime>
#include <vector>
#include <catch.hpp>


SCENARIO("file_system_depth_device is used to debounce the rate at which frames can be acquired", "[seng499][depth_device][file_system_depth_device]") {
	
	GIVEN("30 fps") {
		
		// A fake implementatio of a file_system_depth_device for testing debouncing
		class fake_file_system_depth_device: public seng499::file_system_depth_device::file_system_depth_device {

			public:
			
				fake_file_system_depth_device(int max_fps): file_system_depth_device(max_fps) { }
				
				std::size_t width() const noexcept { return 0; }
				std::size_t height() const noexcept { return 0; }
				Eigen::Matrix3f k() const noexcept { return Eigen::Matrix3f::Zero(); }

				std::vector<float> get_file_system_frame(std::vector<float> vec) const {
					return vec;
				}

		};
		
		std::vector<float> frame;
		int fps = 30;
		
		fake_file_system_depth_device fsdd(fps);
		
		WHEN("The () operator is invoked in succession") {
			
			fsdd(frame);
			clock_t start = clock();
			fsdd(frame);
			clock_t delta = clock() - start;
			
			THEN("At least 1/30 seconds occurs before the second invocation returns") {
			
				CHECK(((float)delta/CLOCKS_PER_SEC) > (1.0/30.0));
			
			}
			
			
		}
	
	}
	
	
}