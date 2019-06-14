import numpy as np
from abc import ABC, abstractmethod

class Shape(ABC):

    """ Parent class for physical objects in RampWorld """

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
        """
        :returns: the volume the shape holds
        :rtype: float
        """
        pass

    @property
    @abstractmethod
    def dimensions(self):
        pass


    # ---------------- Properties -----------------#

    @property
    def mass(self):
        """
        :returns: volume * density
        :rtype: float
        """
        return self.volume * self.density

    @property
    def appearance(self):
        return self._appearance

    @property
    def density(self):
        return self._density


    @property
    def friction(self):
        return self._friction

    @property
    def position(self):
        return self._pos

    @property
    def orientation(self):
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
        """
        :returns: a json friendly dictionary representation
        :rtype: dict
        """
        d = {}
        d['appearance'] = self.appearance
        d['shape'] = self.shape
        d['density'] = self.density
        d['dims'] = self.dimensions
        d['mass'] = self.mass
        d['friction'] = self.friction
        d['position'] = self.position
        d['orientation'] = self.orientation
        return d
