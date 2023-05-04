import numpy as np
import time
from utils.array import normalize


def compute_vertex_normals(v, F):
    # Ref: https://github.com/isl-org/Open3D/blob/master/cpp/open3d/geometry/TriangleMesh.cpp#L146

    ttt = time.process_time()
    v0 = v[F[:, 0]]; v1 = v[F[:, 1]]; v2 = v[F[:, 2]]

    e01 = v1 - v0; e02 = v2 - v0


    n_F = np.cross(e01, e02)



    n_F = normalize(n_F)



    n = np.zeros_like(v)



    n[F[:, 0]] += n_F
    n[F[:, 1]] += n_F
    n[F[:, 2]] += n_F




    n = normalize(n)

    return n