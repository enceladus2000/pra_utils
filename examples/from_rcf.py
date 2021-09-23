# hack to import ros_acoustics module
import pathlib
import sys
parent_dir = pathlib.Path(sys.argv[0]).\
				parent.absolute().\
				parent.absolute().\
				__str__()
sys.path.append(parent_dir)

from pra_utils.core import ComplexRoom
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

path_to_rcf = 'test/data/t_pipe.rcf'

room = ComplexRoom.from_rcf(path_to_rcf, fs=18000, 
					# material=pra.Material(0.4, 0.1),
				)
room.plot()
plt.show()

# TODO: add source inside lol
# room.add_source((5,1,1))
# room.add_microphone((7,1,1))

# room.plot_rir()
# plt.show()

for w in room.walls:
	print(w.absorption[0], w.scatter[0])