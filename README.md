# Pyroomacoustics Utilities

This is a python package that adds some extra functionality to the pyroomacoustics package, specifically in favour of the upcoming ROS acoustics package. 

## Installation

1. Clone this repo
2. Installed prerequisites:

    `pip3 install -r requirements.txt`

3. Run tests (manually for now, haven't set up unit tests yet), to check if it works.
4. To use in this package in your project, export the path to this package to PYTHONPATH:

    `export PYTHONPATH=/path/to/pra_utils:$PYTHONPATH`

## Added Features

All features are encapsulated in `ComplexRoom` in `core.py`, a child class of pyroomacoustics.Room. The notable features are as follows:

1. Improved room plotting:
    - Figure equal aspect ratio
    - Shows walls normals.
2. `from_stl()`, `save_stl`: Creates/saves a room from/to an STL mesh file.
3. `make_polygon()`: Makes a polygonal room. Can specify centre and rotation euler angles. 
4. Obstacles can be added inside rooms. These are basically just reflecting surfaces _inside_ the parent room. To use, first declare a ComplexRoom with `reversed_normals=True`, then add it to a normal ComplexRoom using `add_obstacle()`. See the tests for details.