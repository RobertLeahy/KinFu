#include <kinfu/camera.hpp>


#include <Eigen/Dense>
#include <cmath>
#include <catch.hpp>


SCENARIO("kinfu::to_pixel translates from camera to pixel coordinates and kinfu::to_camera translates from pixel to camera coordinates","[kinfu][to_pixel][to_camera]") {

	GIVEN("A camera calibration matrix, a pixel, and a depth") {

		Eigen::Matrix3f k;
		k <<	585.0f, 0.0f, 320.0f,
				0.0f, -585.0f, 240.0f,
				0.0f, 0.0f, 1.0f;
		
		Eigen::Vector2i pixel(0,0);
		float depth(1.2f);

		WHEN("kinfu::to_camera is invoked thereupon") {

			auto cam=kinfu::to_camera(pixel,depth,k);

			THEN("The result is correct") {

				CHECK(cam(2)==Approx(1.2f));
				CHECK(cam(0)<0.0f);
				CHECK(cam(1)>0.0f);
				CHECK(std::abs(cam(0))>std::abs(cam(1)));

				AND_WHEN("The result is passed to kinfu::to_pixel") {

					auto pair=kinfu::to_pixel(cam,k);

					THEN("The original pixel is retrieved") {

						CHECK(pair.first(0)==pixel(0));
						CHECK(pair.first(1)==pixel(1));

					}

					THEN("The original depth is retrieved") {

						CHECK(pair.second==Approx(depth));

					}

				}

			}

		}

	}

}
