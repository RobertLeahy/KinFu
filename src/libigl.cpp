#include <dynfu/libigl.hpp>
#include <igl/copyleft/marching_cubes.h>
#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <igl/writeOFF.h>


namespace dynfu {


	namespace libigl {


		void viewer (const Eigen::MatrixXd & vertices, const Eigen::MatrixXi & faces) {

			igl::viewer::Viewer viewer;
			viewer.data.set_mesh(vertices,faces);
			viewer.launch();

		}


		void marching_cubes (const Eigen::VectorXd & scalar_field, const Eigen::MatrixXd & field_points, std::size_t x, std::size_t y, std::size_t z, Eigen::MatrixXd & vertices, Eigen::MatrixXi & faces) {

			igl::copyleft::marching_cubes(scalar_field,field_points,x,y,z,vertices,faces);

		}


		void writeOFF(const std::string path, Eigen::MatrixXd & vertices, Eigen::MatrixXi & faces) {
		
			igl::writeOFF(path,vertices,faces);

		}


		void readOFF(const std::string path, Eigen::MatrixXd & vertices, Eigen::MatrixXi & faces) {
		
			igl::readOFF(path,vertices,faces);

		}


	}


}
