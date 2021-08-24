# hack to import ros_acoustics module
import pathlib
import sys
parent_dir = pathlib.Path(sys.argv[0]).\
				parent.absolute().\
				parent.absolute().\
				parent.absolute().\
				__str__()
sys.path.append(parent_dir)

from ros_acoustics.utils.pra_utils import ComplexRoom
import matplotlib.pyplot as plt

path_to_stl = 'test/data/simple_pipe.stl'

room = ComplexRoom.from_stl(path_to_stl)
room.plot(show_normals={'length':.6})
plt.show()
