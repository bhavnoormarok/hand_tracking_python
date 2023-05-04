from utils.freq_imports import *


def compute_global_trans_from_palm_keypoints(k_data_palm, k_model_palm):
    # use point-to-point registration 
    pcd_k_data_palm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(k_data_palm))
    pcd_k_model_palm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(k_model_palm))
    corres = np.repeat(np.arange(len(k_data_palm))[:, np.newaxis], 2, axis=1)
    corres = o3d.utility.Vector2iVector(corres)
    trans_est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    T_glob = trans_est.compute_transformation(pcd_k_model_palm, pcd_k_data_palm, corres)
    R_glob = T_glob[:3, :3]
    t_glob = T_glob[:3, 3]
    
    return R_glob, t_glob