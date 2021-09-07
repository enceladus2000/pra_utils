from __future__ import annotations

import pprint as pp

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import yaml
from mpl_toolkits import mplot3d as a3
from pyroomacoustics import room
from stl import mesh

from .helpers import Limits, BoundingBox, NormalsType

class ComplexRoom(pra.Room):
	"""Extends functionality of pra.Room"""

	# interactive plot fig and axis
	pi_fig = None
	pi_ax = None

	def __init__(self, 
		walls, 
		max_order=4,
		normals_type=NormalsType.none_reversed,
		**kwargs,
	):
		assert isinstance(normals_type, NormalsType)

		super().__init__(walls, max_order=max_order, **kwargs)
		self.normals_type = normals_type
		self._bounding_box = None
	
	def plot_interactive(self,
		wireframe=False, 
		highlight_wall: int = None,
		interactive = True,
	) -> None:
		"""Function for creating an interactive persistent plot.

		Args:
			wireframe (bool, optional): If true, displays only wireframe. Defaults to False.
			highlight_wall (int, optional): Index of the wall to highlight.
		"""
		firsttime = False
		# create figure if haven't already
		if not self.pi_fig:
			self.pi_fig = plt.figure()
			self.pi_ax = a3.Axes3D(self.pi_fig)
			firsttime = True

		# recreate the figure if not interactive
		if not interactive and not firsttime:
			self.pi_fig = plt.figure()
			self.pi_ax = a3.Axes3D(self.pi_fig)
		else:
			self.pi_ax.clear()

		default_clr = (0.5, 0.5, 0.9) if wireframe is False else (1.0,) * 3
		highlight_clr = (.3, .9, .4)
		edge_clr = (0,0,0)

		# plot the walls
		for w in self.walls:
			p = a3.art3d.Poly3DCollection([w.corners.T], alpha=0.3, lw=1)
			if w.name.split('_')[1] == str(highlight_wall):
				p.set_color(colors.rgb2hex(highlight_clr))
			else:
				p.set_color(colors.rgb2hex(default_clr))
			p.set_edgecolor(colors.rgb2hex(edge_clr))
			self.pi_ax.add_collection3d(p)

	def plot(self, show_normals=False, **kwargs):
		"""Plots room with equal axis constrained to the room's bounding cube."""
		fig, ax = super().plot(**kwargs)
		
		# set equal axis aspect ratio hack
		bounding_cube = self.get_bounding_box().get_bounding_cube()
		ax.set_xlim3d(left=bounding_cube.left, right=bounding_cube.right)
		ax.set_ylim3d(bottom=bounding_cube.left, top=bounding_cube.right)
		ax.set_zlim3d(bottom=bounding_cube.left, top=bounding_cube.right)

		# TODO: centre the room in the plot

		if isinstance(show_normals, dict):
			_show_normals = show_normals.get('display', True)
			_normals_len = show_normals.get('length', 0.2)
		elif isinstance(show_normals, bool):
			_show_normals = show_normals
			_normals_len = 0.2
		else:
			raise TypeError('show_normals paramater should be bool or dict.')
		
		if _show_normals is True:
			# points arranged column-wise in both arrays
			normals_start = np.array(
				[np.mean(wall.corners, axis=1) for wall in self.walls]
			).T
			normals_vec = np.array([wall.normal for wall in self.walls]).T

			ax.quiver(
				normals_start[0], normals_start[1], normals_start[2],
				normals_vec[0], normals_vec[1], normals_vec[2],
				length=_normals_len, normalize=True,
			)

		return fig, ax

	@classmethod
	def from_stl(cls, path_to_stl: str, material: pra.Material = None, scale_factor: float = 1.0) -> ComplexRoom:
		"""Creates a ComplexRoom from an STL mesh file.

		Args:
			path_to_stl (str): file path to .stl
			material (pra.Material, optional): Wall material. Defaults to None.
			scale_factor (float, optional): Room dimensions are multiplied by this. Defaults to 1.0.

		Returns:
			ComplexRoom: Object that has dimensions of STL provided.
		"""
		# TODO: Other room params like fs in args/kwargs?
		material = pra.Material(0.5, None) if material is None else material

		room_mesh = mesh.Mesh.from_file(path_to_stl)
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

		return cls(walls, fs=16000, max_order=4, ray_tracing=False)

	@classmethod
	def from_rcf(cls, path_to_rcf) -> ComplexRoom:
		"""Factory method to create ComplexRoom from a room config file (rcf)

		Args:
			path_to_rcf (str): Path to existing rcf
		"""
		with open(path_to_rcf, 'r') as file:
			# get room dict
			rdin = yaml.load(file, Loader=yaml.FullLoader)
		
		walls = []
		for w in rdin['walls']:
			# TODO: checks for value and type of attributes
			wcorners = np.array(w['corners']).T
			mat = pra.Material(w['material']['absorption'], w['material']['scattering'])

			walls.append(
				pra.wall_factory(
					wcorners,
					mat.energy_absorption['coeffs'],
					mat.scattering['coeffs']
				)
			)

		return cls(walls,
					fs=rdin['fs'],
					max_order=rdin['max_order'],
					air_absorption=rdin['air_absorption'],
					ray_tracing=rdin['ray_tracing']	,
				)

	@classmethod
	def from_bounding_box(cls, 
						bounding_box: BoundingBox, 
						material: pra.Material, 
						reverse_normals: bool = False,
						spacing: float=0,
						**kwargs) -> ComplexRoom:
		"""Creates cuboidal ComplexRoom from specified bounding box.

		Args:
			bounding_box (BoundingBox): 
			material (pra.Material): Wall material
			reverse_normals (bool, optional): Keep true for obstacles. Defaults to False.
			spacing (float, optional): Adds extra amount to each dimension. Defaults to 0.

		Returns:
			ComplexRoom: [description]
		"""
		# aliases for readability
		xl, xr = bounding_box.x.left, bounding_box.x.right
		yl, yr = bounding_box.y.left, bounding_box.y.right
		zl, zr = bounding_box.z.left, bounding_box.z.right

		xl, yl, zl = [v-spacing for v in (xl, yl, zl)]
		xr, yr, zr = [v+spacing for v in (xr, yr, zr)]

		# base of 2D points, is in row-wise, normal facing +z form (assuming RH rule)
		base = np.array([[xl, xr, xr, xl, xl], [yl, yl, yr, yr, yl]]).T
		z_s = np.array([zl, zr, zr, zl])

		walls = []

		for i in range(4):
			# corners is kept row-wise until pra.wall_factory
			corners = np.vstack((
				base[i], base[i], base[i+1], base[i+1],
			))
			corners = np.hstack((
				corners, z_s.reshape(4,1)
			))
			# corners are already in reverse order here (RH rule)
			# hence reverse them unless reverse_normals is true
			corners = corners[::-1] if not reverse_normals else corners

			walls.append(
				pra.wall_factory(
					corners.T,
					material.absorption_coeffs,
					material.scattering_coeffs,
				)
			)

		# according to RH rule, bottom_wall_corners is already reversed i.e. facing inwards
		bottom_wall_corners = np.hstack((
			base[0:4],
			np.full((4,1), zl),
		))[::-1]
		top_wall_corners = np.hstack((
			base[0:4],
			np.full((4,1), zr),
		))

		if reverse_normals:
			bottom_wall_corners = bottom_wall_corners[::-1]
			top_wall_corners = top_wall_corners[::-1]

		walls.append(
			pra.wall_factory(
				bottom_wall_corners.T,
				material.absorption_coeffs,
				material.scattering_coeffs,
			))

		walls.append(
			pra.wall_factory(
				top_wall_corners.T,
				material.absorption_coeffs,
				material.scattering_coeffs,
			)
		)
		normals_type = NormalsType.all_reversed if reverse_normals else NormalsType.none_reversed
		return cls(walls, normals_type=normals_type, **kwargs)

	# def

	def save_rcf(self, path_to_rcf: str) -> None:
		"""Saves room into a room configuration file (rcf) for later use.

		Args:
			path_to_rcf (str): Path to save rcf file.
		"""
		rd = self._create_room_dict()
		with open(path_to_rcf, 'w') as file:
			yaml.dump(rd, file, default_flow_style=False, sort_keys=False)

	def save_stl(self, path_to_stl: str) -> None:
		"""Converts a pyroomacoustics room object into an stl mesh

		Args:
			path_to_stl (string): Path of the outputted stl file
		"""
		num_walls = len(self.walls)
		room_mesh = mesh.Mesh(np.zeros(num_walls*2, dtype=mesh.Mesh.dtype))

		for widx, wall in enumerate(self.walls):
			corners = wall.corners.T
			room_mesh.vectors[2*widx] = np.array([
				corners[0], corners[1], corners[2]
			])
			room_mesh.vectors[2*widx+1] = np.array([
				corners[0], corners[2], corners[3]
			])

		room_mesh.save(path_to_stl)

	def print_details(self):
		"""Nicely prints details ab pra.Room object"""
		pp.pprint(self._create_room_dict, sort_dicts=False)

	def _create_room_dict(self) -> dict:
		"""Returns a dictionary containing various room parameters and
		wall vertices.
		"""
		rd = dict()     # room_dict
		rd['ray_tracing'] = self.simulator_state['rt_needed']
		rd['air_absorption'] = self.simulator_state['air_abs_needed']
		rd['max_order'] = self.max_order
		rd['fs'] = self.fs

		rd['walls'] = []
		for widx, wall in enumerate(self.walls):
			wd = dict() # dict for a single wall
			wd['id'] = widx
			wd['material'] = {
				'absorption': float(wall.absorption[0]),
				'scattering': float(wall.scatter[0]) if wall.scatter > 0 else None
			}
			
			# TODO: How to determine normal is reversed
			# wd['reverse_normal'] = False

			corners = wall.corners.T.tolist()
			wd['corners'] = []
			for corner in corners:
				wd['corners'].append(corner)

			rd['walls'].append(wd)

		return rd

	def spatial_transform(self, translate, rpy=[0,0,0]):
		"""Transform room's wall's coordinates by a translation and euler rotation.

		Args:
			translate (list-like of shape(3,)): Vector that describes translation.
			rpy (list-like of shape(3,), optional): Euler angles. Defaults to [0,0,0].

		Raises:
			NotImplementedError: Haven't implemented rpy yet.
		"""

		# TODO: implement rpy and check if room is mix
		if list(rpy) != [0,0,0]:
			raise NotImplementedError
		
		n_walls = []
		for w in self.walls:
			n_walls.append(
				pra.wall_factory(
					w.corners + np.array([translate]).T,
					w.absorption,
					w.scatter,
					w.name,
				)
			)

		self._reinit_with_new_walls(n_walls)

	## Room utils
	@classmethod
	def make_polygon(cls, 
		material: pra.Material,
		centre, 
		radius,
		height,
		N=3,
		rpy=[0,0,0],
		reverse_normals=False,
	) -> ComplexRoom:
		"""Creates polygonal ComplexRoom

		Args:
			material (pra.Material): Material of walls
			centre (arraylike float[3]): Coordinates of centre.
			radius (float): Distance from centre to unextruded polygon's vertices.	
			height (float): Height of extruded polygon.
			N (int, optional): Number of sides. Defaults to 3.
			rpy (arraylike float[3], optional): Roll, pitch, yaw. Defaults to [0,0,0].
			reverse_normals (bool, optional): Keep True for obstacles. Defaults to False.

		Returns:
			ComplexRoom: Poygonal ComplexRoom.
		"""
		wall_faces = ComplexRoom._make_polygon_walls(centre, radius, height, N, rpy, reverse_normals)
		walls = ComplexRoom._construct_walls(wall_faces, material)
		normals_type = NormalsType.all_reversed if reverse_normals else NormalsType.none_reversed
		return cls(walls, normals_type=normals_type)

	def add_obstacle(self, obstacle: ComplexRoom) -> None:
		"""Add another room as an 'obstacle' in a larger room.

		Args:
			obstacle (ComplexRoom): Room to add inside self

		Raises:
			NotImplementedError: TODO: Need to reverse normals of obstacle if isn't already
		"""
		if self.normals_type is not NormalsType.none_reversed:
			raise NotImplementedError('Parent room must have unreversed normals.')

		self.normals_type = NormalsType.mix
		
		walls = self.walls 
		if obstacle.normals_type is NormalsType.none_reversed:
			# reverse wall normals by remaking walls wherein corners are reversed
			for wall in obstacle.walls:
				corners = np.flip(wall.corners, axis=1)
				walls.append(
					pra.wall_factory(
						corners,
						wall.absorption,
						wall.scatter,
						wall.name,
					)
				)
		elif obstacle.normals_type is NormalsType.all_reversed:
			walls += obstacle.walls
		else:
			raise NotImplementedError('Cannot add obstacle with mixed normals to room.')

		self._reinit_with_new_walls(walls, NormalsType.mix) # TODO: reinit with other params?

	def _reinit_with_new_walls(self, n_walls, n_normals_type=None):
		"""Re-initialise the room object with a new set of walls. The rest of 
			the parameters remain the same.

		Args:
			n_walls (list of pra.Wall): List of pra Wall objects
			n_normals_type (NormalsType or None): If none, self.normals_type is used
		"""
		self.__init__(
			n_walls,
			fs=self.fs,
			max_order=self.max_order,
			air_absorption=self.air_absorption,
			ray_tracing=self.ray_tracing,
			normals_type=self.normals_type if n_normals_type is None else n_normals_type
		)

	def _calc_bounding_box(self):
		"""Protected method that determines the bounding box of the room and stores it in self._bounding_box."""
		self._bounding_box = BoundingBox()

		for w in self.walls:
			xmax, xmin = w.corners[0].max(), w.corners[0].min()
			ymax, ymin = w.corners[1].max(), w.corners[1].min()
			zmax, zmin = w.corners[2].max(), w.corners[2].min()
			
			self._bounding_box.x.update(xmin, xmax)
			self._bounding_box.y.update(ymin, ymax)
			self._bounding_box.z.update(zmin, zmax)

	def get_bounding_box(self) -> BoundingBox:
		"""Returns bounding box of room."""
		if self._bounding_box is None:
			self._calc_bounding_box()
		return self._bounding_box
	
	@staticmethod
	def _make_polygon_walls(centre, radius, height, N=3, rpy=[0,0,0], reverse_normals=False) -> ComplexRoom:
		"""Create an extruded polygonal room

		Args:
			centre (array-like): centre of mass of polygon
			radius (float): distance from centre to side edge
			height (float): height of extrusion
			N (int, optional): Number of sides. Defaults to 3.
			rpy (array-like, optional): Roll, pitch and yaw (in that order). Defaults to [0,0,0].
			reverse_normals (bool, optional): If true, normals point inward. Keep true for obstacles, false for rooms.
				Defaults to False.
		"""
		lower_points = []
		upper_points = []
		
		for n in range(N):
			x = radius * np.cos(2*np.pi*n/N)
			y = radius * np.sin(2*np.pi*n/N)

			lower_points.append(np.array([x, y, -height/2]))
			upper_points.append(np.array([x, y, height/2]))

		# do rotation and translation
		cr, cp, cy = np.cos(rpy)
		sr, sp, sy = np.sin(rpy)
		Rx = np.array([
			[1, 0, 0],
			[0, cr, -sr],
			[0, sr, cr]
		]).T
		Ry = np.array([
			[cp, 0, sp],
			[0, 1, 0],
			[-sp, 0, cp]
		]).T
		Rz = np.array([
			[cy, -sy, 0],
			[sy, cy, 0],
			[0, 0, 1]
		]).T
		lower_points = np.array(lower_points) @ Rx @ Ry @ Rz + np.array(centre)
		upper_points = np.array(upper_points) @ Rx @ Ry @ Rz + np.array(centre)

		walls = []
		# add side walls
		for i in range(N-1):
			wall = np.array([
					lower_points[i], lower_points[i+1],
					upper_points[i+1], upper_points[i]
				])
			wall = wall[::-1] if reverse_normals else wall
			walls.append(wall)

		# last side wall
		wall = np.array([
					lower_points[N-1], lower_points[0],
					upper_points[0], upper_points[N-1] 
				])
		wall = wall[::-1] if reverse_normals else wall
		walls.append(wall)

		# lower_points was already created in the reversed order
		# since by RH rule, normal is pointing inward
		if reverse_normals:
			upper_points = upper_points[::-1]
		else:
			lower_points = lower_points[::-1]

		# lower and upper walls
		walls.append(np.array(lower_points))
		walls.append(np.array(upper_points))

		return walls

	@staticmethod
	def _construct_walls(wall_faces, materials):
		"""Returns a list of Wall objects that can be used in the Room constructor

		Args:
			wall_faces (array/list of Nx3 numpy 2D matrices): Same as the output of make_polygon. 
				Each element of this is a 2D matrix representing a wall, each row is a coordinate
				of a vertex.
			material (pra.Material): Material of the wall

		Returns:
			list of Wall objects: Can be directly use in Room constructor
		"""
		material_list = False 
		if isinstance(materials, list):
			if len(materials) != len(wall_faces):
				raise TypeError("list of materials should be same length as obstacle_faces")
			else:
				material_list = True 
		elif not isinstance(materials, pra.Material):
			raise TypeError("materials should be pra.Material or list of pra.Material")

		walls = []
		for i in range(len(wall_faces)):
			m = materials[i] if material_list else materials
			walls.append(
				pra.wall_factory(
					wall_faces[i].T, 
					m.energy_absorption["coeffs"], 
					m.scattering["coeffs"]
				)
			)

		return walls 

# end of ComplexRoom class

def fix_plt_axs(plt, xylims, zlims):
	plt.gca().set_xlim3d(left=xylims[0], right=xylims[1])
	plt.gca().set_ylim3d(bottom=xylims[0], top=xylims[1])
	plt.gca().set_zlim3d(bottom=zlims[0], top=zlims[1])
