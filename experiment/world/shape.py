from abc import ABC, abstractmethod

class Shape(ABC):

    """ Parent class for physical objects in RampWorld """

    def __init__(self, appearance, dims, density, friction):
        self.appearance = appearance
        self.dims = dims
        self.density = density
        self.friction = friction

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
    def dimensions(self):
        return self._dimensions

    @property
    def friction(self):
        return self._friction


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

    @dimensions.setter
    def dimensions(self, value):
        v = np.asarray(value)
        if v.size != 3 and v.size != 1:
            raise ValueError("Scale must represent xyz")
        self._dimensions = v

    # ----------------   Methods   -----------------#
    def serialize(self):
        """
        :returns: a json friendly dictionary representation
        :rtype: dict
        """
        d = {}
        d["Material"] = self.material
        d["Shape"] = self.shape
        d["Density"] = self.density
        d["Scaling"] = list(self.dimensions)
        d["Mass"] = self.mass
        d["Friction"] = self.friction
        return d
