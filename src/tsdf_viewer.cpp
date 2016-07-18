#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/kinect_fusion_opencl_update_reconstruction_pipeline_block.hpp>
#include <dynfu/libigl.hpp>
#include <dynfu/msrc_file_system_depth_device.hpp>
#include <dynfu/opencl_depth_device.hpp>
#include <dynfu/path.hpp>
#include <Eigen/Dense>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>

static dynfu::filesystem::path cl_path () {

    dynfu::filesystem::path retr(dynfu::current_executable_parent_path());
    retr/="..";
    retr/="cl";

    return retr;

}

void main_impl (int, char **) {

    boost::compute::device dev(boost::compute::system::default_device());
    boost::compute::context ctx(dev);
    boost::compute::command_queue q(ctx,dev);
    std::size_t width(640);
    std::size_t height(480);
    Eigen::Matrix3f k;
    Eigen::Matrix4f t_g_k;
    dynfu::file_system_opencl_program_factory fsopf(cl_path(),ctx);
    std::size_t tsdf_width(512);
    std::size_t tsdf_height(512);
    std::size_t tsdf_depth(512);
    float tsdf_extent(3.0);


    //needs to be a valid k matrix for a meaningful test
    k << 585.0f, 0.0f, 320.0f,
        0.0f, 585.0f, 240.0f,
        0.0f, 0.0f, 1.0f;

    //needs to be a valid t_g_k for meaningful test (?)
    t_g_k = Eigen::Matrix4f::Identity();
    t_g_k(0,3) = 1.5f;
    t_g_k(1,3) = 1.5f;
    t_g_k(2,3) = 1.5f;
    dynfu::cpu_pipeline_value<Eigen::Matrix4f> t_g_k_pv;
    t_g_k_pv.emplace(t_g_k);


    dynfu::kinect_fusion_opencl_update_reconstruction_pipeline_block kfourpb(q, fsopf, 5.0f, tsdf_width, tsdf_height, tsdf_depth);
    dynfu::filesystem::path pp(dynfu::current_executable_parent_path());
    dynfu::msrc_file_system_depth_device_frame_factory ff;
    dynfu::msrc_file_system_depth_device_filter f;
    dynfu::file_system_depth_device ddi(pp/".."/"data/test/tsdf_viewer/",ff,&f);
    dynfu::opencl_depth_device dd(ddi,q);

    std::cout << "Generating TSDF..." << std::endl;
	auto && t=kfourpb(*dd(), width, height, k, t_g_k_pv);
	auto && ts = t.buffer->get();
    std::cout << "Finished generating TSDF" << std::endl;

    float px,py,pz;
    Eigen::MatrixXi SF;
    Eigen::MatrixXd SV;
    Eigen::VectorXd S_(tsdf_depth*tsdf_height*tsdf_width);
    Eigen::MatrixXd GV(tsdf_width*tsdf_height*tsdf_depth,3);
    std::size_t abs_idx(0);

    for(std::size_t zi(0); zi < tsdf_depth; ++zi) {

        pz = (float(zi)+0.5f) * tsdf_extent/tsdf_depth;

        for(std::size_t yi(0); yi < tsdf_height; ++yi) {

            py = (float(yi)+0.5f) * tsdf_extent/tsdf_height;

            for(std::size_t xi(0); xi < tsdf_width; ++xi) {

                px = (float(xi)+0.5f) * tsdf_extent/tsdf_width;
                
                std::size_t vox_idx(xi + tsdf_width * (yi + tsdf_height * zi));

                GV.row(vox_idx) = Eigen::RowVector3d(px,py,pz);
                S_(abs_idx) = ts[vox_idx];
                abs_idx++;

            }

        }

        std::cout << "Export Progress: "  << (zi + 1) << " / " << tsdf_depth << std::endl;

    }

    std::cout << "Running Marching Cubes..." << std::endl;

    dynfu::libigl::marching_cubes(S_,GV,tsdf_width,tsdf_height,tsdf_depth,SV,SF);
    dynfu::libigl::viewer(SV,SF);

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
