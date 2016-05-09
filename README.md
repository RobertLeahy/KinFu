# SENG 499 Summer 2016 {#mainpage}
[![Stories in Ready](https://badge.waffle.io/RobertLeahy/SENG499.png?label=ready&title=Ready)](http://waffle.io/RobertLeahy/SENG499)
[![Build Status](https://travis-ci.org/RobertLeahy/SENG499.svg?branch=master)](https://travis-ci.org/RobertLeahy/SENG499)

## Introduction
This repository serves as the home for the course [SENG 499](http://www.ece.uvic.ca/~elec499/) offered at the University of Victoria in the Summer of 2016. The proposed project is to implement a system capabale of generating and continuously updating triangular meshes through the use of a RGBD camera, such as the Microsoft Kinect. Similar work has been done in the following papers.
* [KinectFusion: Real-Time Dense Surface Mapping and Tracking](http://homes.cs.washington.edu/~newcombe/papers/newcombe_etal_ismar2011.pdf)
* [DynamicFusion: Reconstruction and Tracking of Non-rigid Scenes in Real-Time](http://grail.cs.washington.edu/projects/dynamicfusion/papers/DynamicFusion.pdf)

## Building

### Platforms
Tested on
* Windows 10 (MinGW64)
* Ubuntu 14.04 and 16.04

May Work on
* OS/X

### Dependencies
* [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* Get the header-only [Catch](https://github.com/philsquared/Catch) by copying [catch.hpp](https://raw.githubusercontent.com/philsquared/Catch/master/single_include/catch.hpp) to SENG499/include/catch.hpp
* Requires g++-5

#### Directions for Linux
```
git clone https://github.com/RobertLeahy/SENG499.git
cd SENG499
mkdir -p include
wget -O include/catch.hpp https://raw.githubusercontent.com/philsquared/Catch/master/single_include/catch.hpp
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install g++-5 libeigen3-dev

```

### Compile
Once the dependencies are satisfied...
```
cd SENG499
make
```

## Links
* Project Management - [waffle.io](https://waffle.io/RobertLeahy/SENG499)
* Continuous Integration - [travis-ci](https://travis-ci.org/RobertLeahy/SENG499/)

## Contributing
Currently, no outside contributions are being accepted as the coursework is required to be our own.

## License
See the file LICENSE in the root of the repository.

