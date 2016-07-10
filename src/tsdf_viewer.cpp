#include <cstdlib>
#include <exception>
#include <iostream>


void main_impl (int, char **) {

    

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
