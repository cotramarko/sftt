import time
import torch
import numpy as np
from my_map import Map
import utils_np


def normal_pdf(x, mu, cov):
    a = 1 / torch.sqrt(2 * 3.14 * cov)
    b = torch.exp(-torch.pow(x - mu, 2) / (2 * cov))

    return a * b


def valid_point(p, rects):
    # rects = torch.Tensor(rects).type(torch.DoubleTensor)
#    rects = rects.type(torch.DoubleTensor)

    A = rects[:, :2]
    B = torch.cat((rects[:, 0, None], rects[:, 1, None] + rects[:, 3, None]), dim=1)
    D = torch.cat((rects[:, 0, None] + rects[:, 2, None], rects[:, 1, None]), dim=1)  # Mx2

    # TODO: Remove this later on
    M = p

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


def distance_to_object(x, y, phi, dists, rects):
    '''
    All inputs are pytorch Tensors
    '''
    base_point = torch.cat((x[:, None], y[:, None]), dim=1).view(1, -1, 2)
    d_vec = torch.cat((torch.cos(phi)[:, None], torch.sin(phi)[:, None]), dim=1).view(1, -1, 2)

#    dists = torch.arange(0, 15, 0.01).view(-1, 1, 1).type(torch.FloatTensor).cuda()

    ray = base_point + dists * d_vec
    (D, N, _) = ray.shape
    all_points = ray.view(-1, 2)

#    rects = torch.Tensor([map_object.room] + map_object.boundry).type(torch.FloatTensor).cuda()

    idx = ~valid_point(all_points, rects)
    idx = idx.view(D, N)

    invalid_dists = dists.view(-1, 1) * idx.type(torch.cuda.FloatTensor)
    invalid_dists = invalid_dists.view(-1)
    invalid_dists[invalid_dists == 0] = 999999
    invalid_dists = invalid_dists.view(D, N)

    z, idx_z = torch.min(invalid_dists, dim=0)

    # Ignore this since it takes up memory
#    ray = ray.cpu()
#    ray_hits = ray[idx_z.cpu(), np.arange(N), :]
#    ray_hits = None
#    ray = ray.permute(1, 0, 2)

    return None, None, z


if __name__ == '__main__':
    # Test normal_pdf function
    s = torch.Tensor([2]).type(torch.FloatTensor).cuda()
    start = time.time()

    m = torch.Tensor([1]).type(torch.FloatTensor).cuda()
    print('\tSend scalar to gpu time:', time.time() - start)

    x = torch.from_numpy(np.arange(0, 5)).view(-1, 1).type(torch.FloatTensor).cuda()
    print(x, m, s)

    r = normal_pdf(x, m, s)
    print(r)

    N = 1000
    x = np.random.uniform(0, 10, N)
    y = np.random.uniform(0, 10, N)
    phi = np.random.uniform(0, 2 * np.pi, N)

#    x = np.array([0, 1, 2])
#    y = np.array([1, 2, 3])
#    phi = np.array([0, np.pi / 3, np.pi / 2])

    my_map = Map()
    start = time.time()
    (rays, hits, d_pred) = utils_np.distance_to_object(x, y, phi, my_map.room, my_map.boundry)
    print('\tTotal time for numpy:', time.time() - start)

    start0 = time.time()

    use_async = False
    x = torch.from_numpy(x).type(torch.FloatTensor).cuda(async=use_async)
    y = torch.from_numpy(y).type(torch.FloatTensor).cuda(async=use_async)
    phi = torch.from_numpy(phi).type(torch.FloatTensor).cuda(async=use_async)
    dists = torch.arange(0, 15, 0.01).view(-1, 1, 1).type(torch.FloatTensor).cuda(async=use_async)
    rects = torch.Tensor([my_map.room] + my_map.boundry).type(torch.FloatTensor).cuda(async=use_async)

    print('Time taken to move data to gpu:', time.time() - start0)


    start = time.time()
    (rays, hits, d_pred) = distance_to_object(x, y, phi, dists, rects)
    print('Time taken for gpu torch:', time.time() - start)

    start = time.time()
#    r = rays.cpu()
    d_pred = d_pred.cpu()
    print('Time taken to move from gpu to cpu:', time.time() - start)
    
    print('\tTotal time for pytorch:', time.time() - start0)

