# hack to import ros_acoustics module
import pathlib
import sys

from numpy.lib.utils import source
parent_dir = pathlib.Path(sys.argv[0]).\
				parent.absolute().\
				parent.absolute().\
				__str__()
sys.path.append(parent_dir)
print(parent_dir)

"""Tests the make_polygon factory method and plot using show_normals=True"""

from pra_utils.core import ComplexRoom
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2, suppress=True)

# rpy = np.pi * np.random.rand(3) - np.pi/2
# rpy = [np.pi/2, np.pi/4, np.pi/3]
rpy = [0,0,0]
print(rpy)

room_material = pra.Material(0.8, None)
room = ComplexRoom.make_polygon(
		material=room_material,
		centre=[0,0,0], 
		radius=5, 
		height=2.3, 
		N=4, 
		rpy=rpy,
		reverse_normals=False,
		fs=14500,
	)

# source_pos = [1,0,.3]
# room.add_source(source_pos)
# room.add_microphone([0,0,0.5])

print(room.fs)

# print('Volume: ', room.get_volume())
# for w in room.walls:
# 	print(w.normal / np.linalg.norm(w.normal), w.corners[:,0], w.area())
# 	print(w.corners)

# room.compute_rir()

# plot room
room.plot(show_normals={'length':1.}, img_order=1)
plt.show()
# room.compute_rir
# plot rir
# room.plot_rir()
# plt.show()
