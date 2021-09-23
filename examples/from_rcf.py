from pra_utils.core import ComplexRoom
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

# switch to parent dir of script
from pathlib import Path
import os
parent_dir = Path(__file__).parent
os.chdir(parent_dir)

path_to_rcf = 'data/rcf/t_pipe.rcf'

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