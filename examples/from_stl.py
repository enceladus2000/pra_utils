from pra_utils.core import ComplexRoom
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# switch to parent dir of script
from pathlib import Path
import os
parent_dir = Path(__file__).parent
os.chdir(parent_dir)

path_to_stl = 'data/mesh/simple_pipe.stl'

room_material = pra.Material(0.5, None)
room = ComplexRoom.from_stl(path_to_stl, room_material, reverse_normals=False, 
		scale_factor=3., fs=15500, max_order=3,)

room.add_source([1, 2.5, 2.5])
room.add_microphone([3, 2.5, 2.5])
room.compute_rir()
room.plot_rir()

print(room.fs)
room.plot(show_normals={'length':.6})
plt.show()
