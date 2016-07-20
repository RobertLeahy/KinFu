#include <dynfu/camera.hpp>
#include <cmath>
#include <utility>


namespace dynfu {


	std::pair<Eigen::Vector2i,float> to_pixel (Eigen::Vector3f camera, Eigen::Matrix3f k) noexcept {

		Eigen::Vector3f pixel(k*camera);
		Eigen::Vector2i retr(std::round(pixel(0)/pixel(2)),std::round(pixel(1)/pixel(2)));
		return std::make_pair(retr,pixel(2));

	}


	Eigen::Vector3f to_camera (Eigen::Vector2i pixel, float depth, Eigen::Matrix3f k) noexcept {

		Eigen::Vector3f retr(pixel(0),pixel(1),1.0f);
		retr=k.inverse()*retr;;
		retr*=depth;

		return retr;

	}


}
