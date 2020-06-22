from rbw.worlds import MarbleWorld

class BilliardWorld(MarbleWorld):

    def __init__(self, table_dims, table_phys):
        super().__init__(table_dims, table_phys = table_phys)
        self.init_vel = {}

    def add_object(self, name, obj, x, y,
                   force = None, vel = None):
        """ Places objects on the table
        Objects will be placed directly on the table's surface.
        Parameters
        ----------
        name : str
        The key to reference the object
        obj : ``rbw.shapes.Shape``
        Object to add to table
        x : `float`
        The x location of the object.
        y : `float`
        The y location of the object.
        n : `int`
        Which step to place the object.
        """
        super().add_object(name, obj, x, y, force = force)
        if not (vel is None):
            self.init_vel[name] = vel
