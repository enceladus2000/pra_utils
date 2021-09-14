import os
import xml.etree.ElementTree as et
from typing import Text

import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from stl import mesh

from core import BoundingBox, ComplexRoom, Limits


class SDFConverter:

	def __init__(self, sdf_path, use_geometry='visual'):
		assert use_geometry in ('visual', 'collision')
		self.sdf_path = sdf_path
		self.use_geometry = use_geometry
		self.mesh_paths = dict()

	def add_mesh_path(self, model, link, path):
		if not os.path.isfile(path):
			raise ValueError('Path to mesh resource invalid.')

		key = f'{model}.{link}'
		self.mesh_paths[key] = path

	def convert(self):
		sdftree = et.parse(self.sdf_path)
		sdfroot = sdftree.getroot()

		# TODO: generally attribs can be xml attribs or children, how to handle both cases?

		walls = []
		for model in sdfroot.iter('model'):
			model_name = model.attrib.get('name')
			model_pose = model.find('pose').text

			for link in model.iter('link'):
				link_name = link.attrib.get('name')

				mesh_path = link.find('visual/geometry/mesh/uri').text
				if not os.path.isfile(mesh_path):
					mesh_path = self.mesh_paths.get(f'{model_name}.{link_name}', None)

				if mesh_path is None:
					print('Warning: skipping this link.')
					continue
				else:
					print('Found mesh at ', mesh_path)

				scale = link.find('visual/geometry/mesh/scale').text
				scale = [float(v) for v in scale.split(' ')][0] # TODO: support for 3 scaling
				
				material = pra.Material(0.5, None) 	# TODO: how to implement material

				walls += self._walls_from_stl(mesh_path, material, scale)
		
		room = ComplexRoom(walls)
		return room

	# TODO: add generator for wall names as argument?
	def _walls_from_stl(self, stl_path: str, material: pra.Material, scale_factor: float = 1.):
		room_mesh = mesh.Mesh.from_file(stl_path)
		ntriang = room_mesh.vectors.shape[0]

		walls = []
		for i in range(ntriang):
			walls.append(
				pra.wall_factory(
					room_mesh.vectors[i].T * scale_factor,
					material.energy_absorption['coeffs'],
					material.scattering['coeffs'],
					name='wall_'+str(i),
				)
			)

		return walls

if __name__ == '__main__':
	sdfc = SDFConverter('data/worlds/simple_pipe.world')
	sdfc.add_mesh_path('simple_pipe', 'base_link', '/home/tanmay/Projects/pra_utils/data/mesh/simple_pipe.stl')
	room = sdfc.convert()
	room.plot()
	plt.show()

	# room.save_rcf('data/rcf/simple_pipe.rcf')

