from rbw.simulation import Sim

class TrackSim(Sim):
    def __init__(self, scene_json, client):
        self.client = client
        self.world = scene_json

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, cid):
        if cid < 0:
            raise ValueError('Client is offline')
        self._client = cid
    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, w):
        self.resetSimulation()
        self.setGravity(0, 0, -10)
        self.make_track(w['track'])
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

    def make_track(self, params):
        obj_id = self.loadURDF(params['source'],
                               )
                               # baseOrientation = [0.707,0, 0, 0.707])
        self.update_obj(obj_id, params)
