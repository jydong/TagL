# view.py
# Jingyan Dong
# CS251
# 2/21/17

import numpy as np
import math

class View:

    def __init__(self):

        self.vrp = np.matrix([0.5,0.5,1]) #center of the view windowl; origin of view reference coordinates
        self.vpn = np.matrix([0,0,-1]) #direction of viewing
        self.vup = np.matrix([0,1,0]) #view up vector
        self.u = np.matrix([-1,0,0]) #x-axis of view reference coordinates
        self.extent = [1,1,1] #size of the bounding box in data space in view reference coordinates
        self.screen = [400,400] #size of the output device window in pixels
        self.offset = [20,20]

    def reset(self):
        self.__init__(self)

        
    def build(self):
        # Generate a translation matrix to move the VRP to the origin
        # and then premultiply the vtm by the translation matrix
        vtm = np.identity(4, float)
        t1 = np.matrix([[1, 0, 0, -self.vrp[0, 0]],
                           [0, 1, 0, -self.vrp[0, 1]],
                           [0, 0, 1, -self.vrp[0, 2]],
                           [0, 0, 0, 1]])
        vtm = t1 * vtm

        tu = np.cross(self.vup,self.vpn)
        tvup = np.cross(self.vpn, self.u)
        tvpn = self.vpn

        # normalize
        tu = self.normalize(tu)
        tvup = self.normalize(tvup)
        tvpn = self.normalize(tvpn)

        self.vup = tvup
        self.vpn = tvpn
        self.u = tu

        # align the axes
        r1 = np.matrix([[tu[0, 0], tu[0, 1], tu[0, 2], 0.0],
                           [tvup[0, 0], tvup[0, 1], tvup[0, 2], 0.0],
                           [tvpn[0, 0], tvpn[0, 1], tvpn[0, 2], 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

        vtm = r1 * vtm

        # Translate the lower left corner of the view space to the origin
        vtm = np.matrix([[1, 0, 0, 0.5 * self.extent[0]],
                         [0, 1, 0, 0.5 * self.extent[1]],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]) * vtm

        # Use the extent and screen size values to scale to the screen
        vtm = np.matrix([[-self.screen[0] / self.extent[0], 0, 0, 0],
                         [0, -self.screen[1] / self.extent[1], 0, 0],
                         [0, 0, 1.0 / self.extent[2], 0 ],
                         [0, 0, 0, 1]]) * vtm

        # Finally, translate the lower left corner to the origin and add the view offset,
        # which gives a little buffer around the top and left edges of the window.
        vtm = np.matrix([[1, 0, 0, self.screen[0] + self.offset[0]],
                         [0, 1, 0, self.screen[1] + self.offset[1]],
                         [0, 0, 1, 0 ],
                         [0, 0, 0, 1]]) * vtm

        return vtm

    def normalize(self, V):
        Vx = V[0, 0]
        Vy = V[0, 1]
        Vz = V[0, 2]
        Vnorm = [Vx, Vy, Vz]
        length = math.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)
        Vnorm[0] = Vx / length
        Vnorm[1] = Vy / length
        Vnorm[2] = Vz / length
        return np.matrix(Vnorm)

    def rotateVRC(self, ag1, ag2):
        view_point = self.vrp + self.vpn * self.extent[2] * 0.5
        view_point = view_point.tolist()[0]

        t1 = np.matrix([[1, 0, 0, view_point[0]],
                        [0, 1, 0, view_point[1]],
                        [0, 0, 1, view_point[2]],
                        [0, 0, 0, 1]])

        Rxyz = np.matrix([np.append(self.u.tolist(), 0),
                          np.append(self.vup.tolist(), 0),
                          np.append(self.vpn.tolist(), 0),
                          [0, 0, 0, 1]])

        # a rotation matrix about the Y axis by the VUP angle,
        r1 = np.matrix([[math.cos(ag1), 0, math.sin(ag1), 0],
                        [0, 1, 0, 0],
                        [-math.sin(ag1), 0, math.cos(ag1), 0],
                        [0, 0, 0, 1]])

        # a rotation matrix about the X axis by the U angle
        r2 = np.matrix([[1, 0, 0, 0],
                        [0, math.cos(ag2), -math.sin(ag2), 0],
                        [0, math.sin(ag2), math.cos(ag2), 0],
                        [0, 0, 0, 1]])

        t2 = np.matrix([[1, 0, 0, -view_point[0]],
                        [0, 1, 0, -view_point[1]],
                        [0, 0, 1, -view_point[2]],
                        [0, 0, 0, 1]])

        tvrc = np.matrix([np.append(self.vrp.tolist(), 1),
                          np.append(self.u.tolist(), 0),
                          np.append(self.vup.tolist(), 0),
                          np.append(self.vpn.tolist(), 0)])

        tvrc = (t2 * Rxyz.T * r2 * r1 * Rxyz * t1 * tvrc.T).T

        self.vrp = tvrc[0, 0:3]
        self.u = self.normalize(tvrc[1, 0:3])
        self.vup = self.normalize(tvrc[2, 0:3])
        self.vpn = self.normalize(tvrc[3, 0:3])


    def clone(self):
        new = View()
        new.vrp = self.vrp.copy()
        new.vpn = self.vpn.copy()
        new.vup = self.vup.copy()
        new.u = self.u.copy()
        new.extent = self.extent[:]
        new.screen = self.screen[:]
        new.offset = self.offset[:]
        return new


if __name__ == "__main__":
    v = View()
    print "------- view transformation matrix ------ "
    print v.build()
    print "\n ------- view transformation matrix of a clone View object ------ "
    print v.clone().build()





