#include <dynfu/libigl.hpp>
#include <igl/copyleft/marching_cubes.h>
#include <igl/viewer/Viewer.h>


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


	}


}
