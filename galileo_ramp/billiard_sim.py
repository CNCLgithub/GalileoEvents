import numpy as np
from rbw.simulation import MarbleSim


class BilliardSim(MarbleSim):

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, w):
        self.resetSimulation()
        self.setGravity(0, 0, -10)
        self.make_table(w['table'])
        init_force = w['init_force']
        init_vel = w['init_vel']
        d = {}
        for obj,data in w['objects'].items():
            d[obj] = self.make_obj(data)
            if obj in w['init_force']:
                f = init_force[obj]
                self.applyExternalForce(d[obj], -1, f, [0,0,0],self.LINK_FRAME)
            if obj in w['init_vel']:
                angular, linear = init_vel[obj]
                self.resetBaseVelocity(d[obj],
                                       angularVelocity = angular,
                                       linearVelocity = linear)
        self._world = d

    def make_table(self, params):
        # Table top
        base_id = self.make_obj(params)

        # table walls
        shape = self.GEOM_BOX

        bounds = np.array(params['dims']) # x,y,z
        delta = 0.01
        exs_small = np.array([delta, bounds[1], bounds[2] * 1.1]) * 0.5
        exs_large = np.array([bounds[0], delta, bounds[2] * 1.1]) * 0.5

        pos_left =  bounds * np.array([-0.5, 0, 0.5])
        wall_left = self.createCollisionShape(shape, halfExtents = exs_small)
        obj_id = self.createMultiBody(baseCollisionShapeIndex = wall_left,
                                      basePosition = pos_left)
        self.update_obj(obj_id, params)

        pos_right =  bounds * np.array([0.5, 0, 0.5]) + np.array([delta, 0, 0])
        wall_right = self.createCollisionShape(shape, halfExtents = exs_small)
        obj_id = self.createMultiBody(baseCollisionShapeIndex = wall_right,
                                      basePosition = pos_right)
        self.update_obj(obj_id, params)

        pos_front =  bounds * np.array([0, 0.5, 0.5])
        wall_front = self.createCollisionShape(shape, halfExtents = exs_large)
        obj_id = self.createMultiBody(baseCollisionShapeIndex = wall_front,
                                      basePosition = pos_front)
        self.update_obj(obj_id, params)

        pos_back =  bounds * np.array([0, -0.5, 0.5])
        wall_back = self.createCollisionShape(shape, halfExtents = exs_large)
        obj_id = self.createMultiBody(baseCollisionShapeIndex = wall_back,
                                      basePosition = pos_back)
        self.update_obj(obj_id, params)
