# SENG 499 Summer 2016
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
* Get the header-only [Catch](https://github.com/philsquared/Catch) by copying [catch.hpp](https://raw.githubusercontent.com/philsquared/Catch/master/single_include/catch.hpp) to SENG499/include/catch.hpp
```
# Directions for Linux

cd SENG499
mkdir -p include
wget -O include/catch.hpp https://raw.githubusercontent.com/philsquared/Catch/master/single_include/catch.hpp
```
* Requires g++-5
```
# Included on Ubuntu 16.04 (?)
# Directions for Linux

sudo apt-get install g++-5
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
MIT License

Copyright (c) 2016 Jordan Heemskerk, Robert Leahy, Jorin Weatherston, Tyler Potter, Charlotte Fedderly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


