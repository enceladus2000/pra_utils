# Pyroomacoustics Utilities

This is a python package that adds some extra functionality to the pyroomacoustics package, specifically in favour of the upcoming ROS acoustics package. 

## Installation

1. Clone this repo into any directory.
2. `cd` to the project root directory and install using

    `pip3 install . `

    Note: For development, it is recommended to install in edit mode:

    `pip3 install -e .`

3. Test the install by importing the pra_utils package:

    `python3 -c 'import pra_utils'`
4. Buy me a beer, thanks.

## Added Features

All features are encapsulated in `ComplexRoom` in `core.py`, a child class of pyroomacoustics.Room. The notable features are as follows:

1. Improved room plotting:
    - Figure equal aspect ratio
    - Shows walls normals.
2. `from_stl()`, `save_stl`: Creates/saves a room from/to an STL mesh file.
3. `make_polygon()`: Makes a polygonal room. Can specify centre and rotation euler angles. 
4. Obstacles can be added inside rooms. These are basically just reflecting surfaces _inside_ the parent room. To use, first declare a ComplexRoom with `reversed_normals=True`, then add it to a normal ComplexRoom using `add_obstacle()`. See the tests for details.