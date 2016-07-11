#include <dynfu/filesystem.hpp>
#include <dynfu/path.hpp>
#include <Eigen/Dense>
#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>


void main_impl (int, char **) {

    //  Get path to bunny.off
    dynfu::filesystem::path p(dynfu::current_executable_parent_path());
    p/="..";
    p/="data/test/tsdf_viewer/bunny.off";

    Eigen::MatrixXd v;
    Eigen::MatrixXi f;
    auto p_str=p.string();
    if (!igl::readOFF(p_str,v,f)) {

        std::ostringstream ss;
        ss << "Failed reading " << p_str;
        throw std::runtime_error(ss.str());

    }

    igl::viewer::Viewer viewer;
    viewer.data.set_mesh(v,f);
    viewer.launch();

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
