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
