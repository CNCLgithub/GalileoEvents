from rbw.simulation import MarbleSim


class BilliardSim(MarbleSim):

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, w):
        super().world = w
        init_vel = w['init_vel']
        d = {}
        for obj,data in w['objects'].items():
            if obj in w['init_vel']:
                angular, linear = init_vel[obj]
                self.resetBaseVelocity(d[obj],
                                       angularVelocity = angular,
                                       linearVelocity = linear)
        self._world = d
