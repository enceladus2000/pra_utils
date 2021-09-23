# hack to import ros_acoustics module
# import pathlib
# import sys
# parent_dir = pathlib.Path(sys.argv[0]).\
# 				parent.absolute().\
# 				parent.absolute().\
# 				__str__()
# sys.path.append(parent_dir)

from pra_utils.core import ComplexRoom
import matplotlib.pyplot as plt
import pyroomacoustics as pra

path_to_stl = 'test/data/simple_pipe.stl'

room_material = pra.Material(0.5, None)
room = ComplexRoom.from_stl(path_to_stl, room_material, reverse_normals=True, 
		scale_factor=3., fs=15500, max_order=3,)

print(room.fs)
room.plot(show_normals={'length':.6})
plt.show()
