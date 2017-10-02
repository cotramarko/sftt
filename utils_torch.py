import torch
import numpy as np
from my_map import valid_point_broad, valid_point
from robot import distance_to_object
from my_map import Map


def valid_point_torch(p, rects):
    rects = torch.Tensor(rects).type(torch.DoubleTensor)
#    rects = rects.type(torch.DoubleTensor)

    A = rects[:, :2]
    B = torch.cat((rects[:, 0, None], rects[:, 1, None] + rects[:, 3, None]), dim=1)
    D = torch.cat((rects[:, 0, None] + rects[:, 2, None], rects[:, 1, None]), dim=1)  # Mx2

    M = torch.from_numpy(p)  # Nx2

    AM = M[:, None, :] - A[None, ...]  # NxMx2 points-by-boundries-by-xy
    AB = B - A  # Mx2 boundries-by-xy
    AD = D - A  # Mx2 boundries-by-xy

    AM_AB = torch.sum(AM * AB, dim=-1)  # NxM
    AM_AD = torch.sum(AM * AD, dim=-1)  # NxM

    AB_AB = torch.sum(AB * AB, dim=-1)  # NxM
    AD_AD = torch.sum(AD * AD, dim=-1)  # NxM

    cond1 = (0 < AM_AB) & (AM_AB < AB_AB)
    cond2 = (0 < AM_AD) & (AM_AD < AD_AD)

    final_cond = cond1 & cond2
    final_cond[:, 0] = (~final_cond[:, 0])

    res = ~(torch.sum(final_cond, dim=1))

    return res





if __name__ == '__main__':
    rects = [(3, 0, 1, 4),
             (7, 0, 3, 3),
             (5, 5, 5, 1),
             (2, 8, 5, 2),
             (1, 5, 2, 2)]

    points = np.array([[1, 1], [2, 2], [3, 3], [3.5, 3]])
    # points = np.random.normal(size=(1000000, 2))

    print(valid_point_torch(points, rects))

    print('numpy')

    print(valid_point_broad(points, rects))
 