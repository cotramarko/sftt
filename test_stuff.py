import numpy as np



def cross(xy0, xy1):
    return xy0[0] * xy1[1] - xy0[1] * xy1[0]



print(cross([-9, -9], [9, 10]))