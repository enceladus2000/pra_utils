# hack to import ros_acoustics module
import pathlib
import sys

from numpy.lib.utils import source
parent_dir = pathlib.Path(sys.argv[0]).\
				parent.absolute().\
				parent.absolute().\
				__str__()
sys.path.append(parent_dir)

from pra_utils.complex_room import ComplexRoom
from pra_utils.helpers import NormalsType
import pyroomacoustics as pra
import matplotlib.pyplot as plt

room_material = pra.Material(0.8, None)
room = ComplexRoom.make_polygon(
		material=room_material,
		centre=[0,0,0], 
		radius=5, 
		height=2, 
		N=3, 
		rpy=[0,0,0],
		reverse_normals=False,
	)

obstacle = ComplexRoom.make_polygon(
		material=room_material,
		centre=[0,0,0], 
		radius=1, 
		height=1, 
		N=4, 
		rpy=[0,0,0],
		reverse_normals=False,
	)
obstacle.spatial_transform([.4,0,.3])
print(obstacle.volume)
print(room.volume)
room.add_obstacle(obstacle)

source_pos = [1,-1,0.]
mic_pos = [1, -1, 1]
room.add_source(source_pos)
room.add_microphone(mic_pos)
room.compute_rir()

print(room.volume)

# plot room
room.plot(show_normals={'length': 0.4}, img_order=1)
plt.show()

# plot rir
room.plot_rir()
plt.show()