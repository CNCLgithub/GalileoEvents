from rbw.simulation import RampSim


class Ball3Sim(RampSim):

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, w):
        self.resetSimulation()
        self.setGravity(0, 0, -10)
        self.make_table(w['table'])
        self.make_obj(w['ramp'])
        init_vel = w['init_vel']
        d = {}
        for obj,data in w['objects'].items():
            d[obj] = self.make_obj(data)
            if obj in w['init_vel']:
                angular, linear = init_vel[obj]
                self.resetBaseVelocity(d[obj],
                                       angularVelocity = angular,
                                       linearVelocity = linear)
        self._world = d
