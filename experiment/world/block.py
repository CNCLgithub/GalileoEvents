import numpy as np
from .shape import Shape


class Block(Shape):

    # ---------------- Properties -----------------#

    @property
    def shape(self):
        """ The name of the shape
        :returns: A shape name
        :rtype: str
        """
        return 'Block'

    @property
    def volume(self):
        """
        Defined as volume of a cuboid.
        :returns: the volume the shape holds
        :rtype: float
        """
        return np.prod(self.dimensions)

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
