# SENG 499 Summer 2016 - DynFu {#mainpage}
[![Stories in Ready](https://badge.waffle.io/RobertLeahy/DynFu.png?label=ready&title=Ready)](http://waffle.io/RobertLeahy/DynFu)
[![Build Status](https://travis-ci.com/RobertLeahy/DynFu.svg?token=E1Ypp9btW9nWJKJqzctp&branch=master)](https://travis-ci.com/RobertLeahy/DynFu)

## Introduction
This repository serves as the home for the course [SENG 499](http://www.ece.uvic.ca/~elec499/) offered at the University of Victoria in the Summer of 2016. The proposed project is to implement a system capabale of generating and continuously updating triangular meshes through the use of a RGBD camera, such as the Microsoft Kinect. Similar work has been done in the following papers.
* [KinectFusion: Real-Time Dense Surface Mapping and Tracking](http://homes.cs.washington.edu/~newcombe/papers/newcombe_etal_ismar2011.pdf)
* [DynamicFusion: Reconstruction and Tracking of Non-rigid Scenes in Real-Time](http://grail.cs.washington.edu/projects/dynamicfusion/papers/DynamicFusion.pdf)

## Building

### Platforms
Tested on
* Windows 10 (MinGW64)
* Ubuntu 14.04 and 16.04
* OS/X It builds with no warnings once the Mac build system is merged, but the tests fail horribly due to some issue we suspect with Boost.Compute & OpenCL on Mac OSX.

### Direct Dependencies
* [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* Get the header-only [Catch](https://github.com/philsquared/Catch) by copying [catch.hpp](https://raw.githubusercontent.com/philsquared/Catch/master/single_include/catch.hpp) to SENG499/include/catch.hpp
* Requires g++-5 and/or g++-6 (future proofing in travis-ci build matrices)
* [OpenCV](http://opencv.org/)
* [Boost](http://www.boost.org/)
* [Boost.Compute](https://github.com/boostorg/compute)

#### Directions for Mac
This includes instructions for installing direct project dependencies and also for build environment dependencies.
After installing all systems, cmake should work in your environment. 

#####Boost
Install Homebrew if not already installed
```
brew install boost
```
#####Boost.Compute
```
git clone https://github.com/boostorg/compute.git
copy headers to project: sudo cp -r ./compute/include/boost/* /usr/local/include/boost
```
#####Catch
```
git clone https://github.com/philsquared/Catch.git
copy headers to project: sudo cp ./Catch/single_include/catch.hpp /usr/local/include
```
#####OpenCV
Download from [here](http://opencv.org/downloads.html)

Extract to location of your choice and open terminal there
```
cmake .
make 
make install
```
#####lippcv
Download from [here](https://sourceforge.net/projects/opencvlibrary/files/3rdparty/ippicv/)

Extract and copy lib/* to /usr/local/lib

#####Eigen3
```
brew install eigen
sudo ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen
```
#####pkg-config
```
brew install pkg-config
```

Note: once #69 is merged in this process should  build the entire project without errors. Warnings are still possible because of symlink issues. #69 contains details.

#### Directions for Windows
Different build system on Windows than Linux and Mac. 

##### Mingw64
Download mingw64 gui installer [here](https://sourceforge.net/projects/mingw/files/latest/download)
install to location without spaces in the path of the directory.
Mingw64 can be used to compile with cmake if necessary as [here](http://stackoverflow.com/questions/4101456/running-cmake-on-windows)

##### Eigen3
Download from [here](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)
Extract and include source headers

#####OpenCV 
Installation instructions for OpenCV include compiling the library

Detailed instructions are available here [here](http://docs.opencv.org/2.4/doc/tutorials/introduction/windows_install/windows_install.html#cpptutwindowsmakeown)

#####Boost
Boost must be built for windows if libraries that are not header only will be included.

Boost windows install instructions can be found [here](http://www.boost.org/doc/libs/1_61_0/more/getting_started/windows.html)

#####Boost.Compute
```
git clone https://github.com/boostorg/compute.git
copy headers to accessible include location in project
```
#####Catch
```
git clone https://github.com/philsquared/Catch.git
copy headers to accessible build location in project
```

#### Directions for Linux
Since we are using travis-ci it is possible that issues can be resolved by referencing the .travis.yml file on the master branch for solution inspirations. 

```
git clone https://github.com/RobertLeahy/DynFu.git
cd DynFu
mkdir -p include
wget -O include/catch.hpp https://raw.githubusercontent.com/philsquared/Catch/master/single_include/catch.hpp
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install g++-5 libeigen3-dev libboost-all-dev libopencv-dev

```

### Compile
Once the dependencies are satisfied...
```
cd DynFu
cmake .
make
```

### Documentation

Available at [http://seng499.rleahy.ca/](http://seng499.rleahy.ca/)

Use Doxygen and GraphViz
```
sudo apt-get install graphviz doxygen
cd SENG499
doxygen
```

## Links
* Project Management - [waffle.io](https://waffle.io/RobertLeahy/DynFu)
* Continuous Integration - [travis-ci](https://travis-ci.org/RobertLeahy/DynFu/)

## Contributing
Currently, no outside contributions are being accepted as the coursework is required to be our own.

## License
See the file LICENSE in the root of the repository.
