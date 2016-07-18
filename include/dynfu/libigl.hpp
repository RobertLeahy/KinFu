/**
 *	\file
 */


#include <Eigen/Dense>
#include <string>

namespace dynfu {


	namespace libigl {


		void viewer (const Eigen::MatrixXd & vertices, const Eigen::MatrixXi & faces);


		void marching_cubes (const Eigen::VectorXd & scalar_field, const Eigen::MatrixXd & field_Points, std::size_t x, std::size_t y, std::size_t z, Eigen::MatrixXd & vertices, Eigen::MatrixXi & faces);


		void writeOFF(const std::string path, Eigen::MatrixXd & vertices, Eigen::MatrixXi & faces);


		void readOFF(const std::string path, Eigen::MatrixXd & vertices, Eigen::MatrixXi & faces);


	}


}
