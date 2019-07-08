import numpy as np
from abc import ABC, abstractmethod

class Shape(ABC):

    """ Parent class for physical objects in :mod:`galileo_ramp.world`

    This class describes the interface for
    physical properties, apearance, and geometry
    of objects. 
    """

    def __init__(self, appearance, dims, density, friction,
                 pos = None, angle = None):
        self.appearance = appearance
        self.dimensions = dims
        self.density = density
        self.friction = friction
        self.position = pos
        self.orientation = angle

    # ---------------- Abstract Methods -----------------#

    @property
    @abstractmethod
    def shape(self):
        """ The name of the shape
        :returns: A shape name
        :rtype: str
        """
        pass

    @property
    @abstractmethod
    def volume(self):
        """ Should be implemented by children.
        :returns: volume in m^3
        :rtype: float
        """
        pass

    @property
    @abstractmethod
    def dimensions(self):
        """ The xyz extremas of a given object type.
        """
        pass


    # ---------------- Properties -----------------#

    @property
    def mass(self):
        """ Returns the mass in units
        :returns: volume * density
        :rtype: float
        """
        return self.volume * self.density

    @property
    def appearance(self):
        """ Returns the appearance descriptor"""
        return self._appearance

    @property
    def density(self):
        """ Returns the density in grams/m^3"""
        return self._density


    @property
    def friction(self):
        """ Returns the friction coefficient used by Bullet3D"""
        return self._friction

    @property
    def position(self):
        """ The XYZ coordinate of the object's COM"""
        return self._pos

    @property
    def orientation(self):
        """ The wxyz quaternion for object's rotation"""
        return self._orien

    # ----------------   setters   -----------------#

    @appearance.setter
    def appearance(self, value):
        """ set the appearance of the shape
        :param value: name of the texture
        :type value: str
        """
        self._appearance = value

    @density.setter
    def density(self, value):
        self._density = value


    @friction.setter
    def friction(self, value):
        self._friction = value

    @position.setter
    def position(self, value):
        if value is None:
            value = [0, 0, 0]
        v = np.array(value)
        if v.size != 3:
            raise ValueError('position must be xyz')
        self._pos = v


    @orientation.setter
    def orientation(self, value):
        if value is None:
            value = [0, 0, 0]
        v = np.array(value)
        if v.size != 3:
            raise ValueError('Orientation must be 3 euler angles')
        self._orien = v

    # ----------------   Methods   -----------------#
    def serialize(self):
        """ Returns a json friendly dictionary representation
        :rtype: dict
        """
        d = {}
        d['appearance'] = self.appearance
        d['shape'] = self.shape
        d['density'] = self.density
        d['dims'] = self.dimensions
        d['volume'] = self.volume
        d['mass'] = self.mass
        d['friction'] = self.friction
        d['position'] = self.position
        d['orientation'] = self.orientation
        return d
