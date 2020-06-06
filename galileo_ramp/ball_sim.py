from rbw.simulation import RampSim


class Ball3Sim(RampSim):


    @world.setter
    def world(self, w):
        self.resetSimulation()
        self.setGravity(0, 0, -10)
        self.make_table(w['table'])
        init_vel = w['init_vel']
        d = {}
        for obj,data in w['objects'].items():
            d[obj] = self.make_obj(data)
            if obj in w['init_vel']:
                linear, angular = init_vel[obj]
                self.resetBaseVelocity(d[obj], linearVelocity = linear,
                                       angularVelocity = angular)
        self._world = d
