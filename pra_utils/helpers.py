from enum import Enum
from dataclasses import dataclass

class NormalsType(Enum):
	none_reversed = False	# for normal rooms, facing outwards
	all_reversed = True		# for obstacles, facing inwards
	mix = 2					# for rooms with obstacles

@dataclass
class Limits:
	left: float = 0.
	right: float = 0.

	def __post_init__(self):
		if self.left > self.right:
			raise ValueError('left must not be more than right.')

	def update(self, min: float, max: float) -> None:
		"""Updates left and right values"""
		assert max >= min
		if min < self.left:
			self.left = min
		if max > self.right:
			self.right = max

	@property
	def mid(self):
		return (self.right - self.left) / 2

@dataclass
class BoundingBox:
	"""Stores the 3D coordinate limits of a figure."""
	x: Limits = Limits()
	y: Limits = Limits()
	z: Limits = Limits()
	
	def get_bounding_cube(self) -> Limits:
		"""Get the 3D cube coordinate limits of a box."""
		l = Limits()
		l.left = min(self.x.left, self.y.left, self.z.left)
		l.right = max(self.x.right, self.y.right, self.z.right)
		return l

	@property
	def centre(self):
		return [self.x.mid, self.y.mid, self.z.mid]
