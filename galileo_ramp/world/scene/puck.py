import numpy as np
from .shape import Shape


class Puck(Shape):

    """Instance of Shape describing elliptical cylinders

    Dimensions represent
    """

    # ---------------- Properties -----------------#

    @property
    def shape(self):
        """ The name of the shape
        :returns: A shape name
        :rtype: str
        """
        return "Block"

    @property
    def volume(self):
        """
        Defined as volume of an elliptical cylinder.
        :returns: the volume the shape holds
        :rtype: float
        """
        return np.prod(self.dimensions)* np.pi * 0.25

    @property
    def dimensions(self):
        return self._dimensions

    # ----------------   setters   -----------------#

    @dimensions.setter
    def dimensions(self, value):
        v = np.asarray(value)
        if v.size != 3:
            raise ValueError('Scale must represent xyz')
        self._dimensions = v
