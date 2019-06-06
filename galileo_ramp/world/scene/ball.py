import numpy as np
from .shape import Shape


class Ball(Shape):

    """Instance of Shape describing spheres
    """

    # ---------------- Properties -----------------#

    @property
    def shape(self):
        """ The name of the shape
        :returns: A shape name
        :rtype: str
        """
        return "Ball"

    @property
    def volume(self):
        """
        Defined as volume of an elliptical cylinder.
        :returns: the volume the shape holds
        :rtype: float
        """
        return (self.dimensions**3) * np.pi * (4.0/3.0)

    @property
    def dimensions(self):
        return self._dimensions
    # ----------------  Setters   -----------------#

    @dimensions.setter
    def dimensions(self, value):
        v = np.asarray(value)
        if  v.size != 1:
            raise ValueError('Scale must represent radius of sphere')
        self._dimensions = v
