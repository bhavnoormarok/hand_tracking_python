import numpy as np
import open3d as o3d

def uvd2xyz(U, fx, fy, cx, cy):
    u, v, d = np.hsplit(U, 3)   # hsplit: handles multiple points
    z = d
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.hstack([x, y, z])


def xyz2uv(X, fx, fy, cx, cy):
    x, y, z = np.hsplit(X, 3)   # hsplit: handles multiple points
    u = fx * x / z + cx
    v = fy * y / z + cy

    return np.hstack([u, v])