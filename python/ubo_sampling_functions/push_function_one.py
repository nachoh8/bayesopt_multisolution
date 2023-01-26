from .push_utils import b2WorldInterface, make_base, create_body, end_effector, run_simulation_once

import numpy as np


class PushReward:
    def __init__(self, use_dir=False, use_gui=True):

        self.use_dir = use_dir

        # domain of this function
        self.xmin = [-5., -5., -10., -10., 2., 0.,  -5.]
        self.xmax = [5., 5., 10., 10., 30., 2.*np.pi, 5.]

        # starting xy locations for the two objects
        self.sxy = (0, 2)
        # goal xy locations for the two objects
        self.gxy = [4, 3.5]

        if not self.use_dir:
            self.xmin[2] = 0
            self.xmax[2] = +5
            self.xmin.pop(3)
            self.xmax.pop(3)
            

        self.use_gui = use_gui

        self.lb = np.array(self.xmin)
        self.ub = np.array(self.xmax)

    @property
    def f_max(self):
        # maximum value of this function
        return np.linalg.norm(np.array(self.gxy) - np.array(self.sxy))
    @property
    def dx(self):
        # dimension of the input
        return self._dx
    
    def __call__(self, argv):
        # returns the reward of pushing two objects with two robots
        rx = float(argv[0])
        ry = float(argv[1])

        if self.use_dir:
            xvel = float(argv[2])
            yvel = float(argv[3])
            simu_steps = int(float(argv[4]) * 10)
            init_angle = float(argv[5]) if self.use_dir else None
            rtor = float(argv[6])

        else:
            vel = float(argv[2])
            simu_steps = int(float(argv[3]) * 10)
            init_angle = float(argv[4]) if self.use_dir else None
            rtor = float(argv[5])

            v = np.array(self.sxy) - np.array((rx, ry))
            v = v/np.linalg.norm(v)
            xvel = v[0] * vel
            yvel = v[1] * vel


        if not self.use_dir:
            init_angle = np.arctan2(self.sxy[0]-ry, self.sxy[1]-rx)
        
        initial_dist = self.f_max

        world = b2WorldInterface(self.use_gui)
        oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size = \
            'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (1, 0.3)

        base = make_base(500, 500, world)
        body = create_body(base, world, 'rectangle', (0.5, 0.5), ofriction, odensity, self.sxy)
        
        robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
        ret1, robot_end = run_simulation_once(world, body, robot, xvel, yvel, \
                                      rtor, simu_steps)

        ret1 = np.linalg.norm(np.array(self.gxy) - ret1)
        
        # If there is no contact, penalize based on distance
        p1 = np.array(robot_end)
        p0 = (rx, ry)        
        distance_to_contact = 0.0
        if initial_dist - ret1 < 1e-15:
            # dist_line = np.linalg.norm(np.cross(p1-p0, p0-np.array(self.sxy)))/np.linalg.norm(p1-p0)

            # distance_to_contact = min([dist_line, 
            #     np.linalg.norm(np.array(self.sxy) - p1),
            #     np.linalg.norm(np.array(self.sxy) - p0)])

            distance_to_contact = np.linalg.norm(np.array(self.sxy) - p1)
        return ret1 + distance_to_contact

def main():
    f = PushReward()
    x = np.random.uniform(f.xmin, f.xmax)
    print('Input = {}'.format(x))
    print('Output = {}'.format(f(x)))


if __name__ == '__main__':
    main()
