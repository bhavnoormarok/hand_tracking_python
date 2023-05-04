from utils.freq_imports import *
from scipy.ndimage import distance_transform_edt


def depth_to_pointcloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    V, U = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    
    X = (U - cx) * depth / fx   # (h, w)
    Y = (V - cy) * depth / fy   # (h, w)
    Z = depth                   # (h, w)

    pcd = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1) # (h*w, 3)
    
    # keep points that are valid depth values i.e., ignore background points
    pcd = pcd[pcd[:, 2] > 0]
    return pcd


def compute_d2m_mean_max(depth_obs, depth_ren, fx, fy, cx, cy):
    # obtain point cloud from depth
    pcd_obs = depth_to_pointcloud(depth_obs, fx, fy, cx, cy)
    pcd_ren = depth_to_pointcloud(depth_ren, fx, fy, cx, cy)
    
    kdtree_ren = KDTree(pcd_ren)
    d, i_pcd_ren = kdtree_ren.query(pcd_obs)
    d2m_mean = np.mean(d)
    d2m_max = np.amax(d)

    return d2m_mean, d2m_max



def compute_m2d_mean_max(depth_obs, depth_ren):
    # a silhouette image is the binary image with 1 outside the hand region and 0 inside
    # at each pixel, find index to the closest pixel with value 1, using Distance Transform
    # the points are represented in image frame where origin is at top left; this is consistent with the perspective projection using intrinsic camera parameters
    D_vu_obs = distance_transform_edt(~(depth_obs > 0), return_distances=True, return_indices=False)  # (H, W) -7 fps
    
    # for all valid points in rendered depth, find distance to silhouette in observed depth
    d = D_vu_obs[depth_ren > 0] # (|valid_ren_depth_points|,)
    m2d_mean = np.mean(d)
    m2d_max = np.amax(d)
    
    return m2d_mean, m2d_max

def compute_k_error_mean(k1, k2):
    return np.mean(np.linalg.norm(k1-k2, axis=1))*1000

def compute_k_error_max(k1, k2):
    return np.amax(np.linalg.norm(k1-k2, axis=1))*1000

