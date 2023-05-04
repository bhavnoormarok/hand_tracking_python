from utils.freq_imports import *
from scipy.ndimage import distance_transform_edt
from utils import plotly_wrapper

ccc = 0
ttt = 0
def furthest_point_downsample_ids(D, n_S):
    # Reference: https://www.programmersought.com/article/8737853003/#12_farthest_point_sample_28
    # D: dense points; S: selected points
    n_D = len(D)


    S_ids = np.zeros(n_S, dtype=np.int32)
    
    to_D_min_dists = np.ones(n_D) * 1e9
    
    # rng = np.random.default_rng(1)
    # curr_furthest_S_id = rng.integers(0, n_D)
    curr_furthest_S_id = 0

    #qwe = time.time()
    for id_in_S in range(n_S):
        S_ids[id_in_S] = curr_furthest_S_id

        s = D[curr_furthest_S_id]
        a = np.ascontiguousarray((D - s).T)
        s_to_D_dists = np.einsum("ij,ij->j", a, a) #np.sum((D - s)**2, axis=1)
        to_D_min_dists = np.minimum(s_to_D_dists, to_D_min_dists)
        #print("trew",np.max(to_D_min_dists))

        curr_furthest_S_id = np.argmax(to_D_min_dists)

    #print(time.time()-qwe,"qwe")

    return S_ids

def depth_to_point_cloud(depth_proc, fx, fy, cx, cy, n_x):
    #wsx = time.time()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(depth_proc.shape[1], depth_proc.shape[0], fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_proc.astype(np.uint16)), intrinsic)    # this function requires depth to be in uint16
    
    # pcd = pcd.paint_uniform_color([0, 0, 0])
    x_dense = np.asarray(pcd.points)

    # downsample point cloud for faster computation
    if len(np.asarray(pcd.points)) > n_x:
        pcd = pcd.voxel_down_sample(0.001)
        # pcd = pcd.paint_uniform_color([0, 0, 1])
        # o3d.visualization.draw_geometries([pcd])
        # exit()
        # scat_x = plotly_wrapper.scatter3d(np.asarray(pcd.points), 3, "red")
        # fig = go.Figure(scat_x)
        # plotly_wrapper.remove_fig_background(fig)
        # plotly_wrapper.update_fig_size(fig, width=1000, height=1000)
        # fig.update_layout(scene=dict(aspectmode='data'))
        # fig.show()
        # exit()
    
    # x_plot = None
    # remove outlier
    if len(np.asarray(pcd.points)) > n_x:
        pcd, ids = pcd.remove_radius_outlier(nb_points=20, radius=0.01)    # remove points that have less than `nb_points` in a sphere of `radius`
        # x_plot = np.asarray(pcd.points)
        # scat_x = plotly_wrapper.scatter3d(np.asarray(pcd.points), 3, "red")
        # fig = go.Figure(scat_x)
        # plotly_wrapper.remove_fig_background(fig)
        # plotly_wrapper.update_fig_size(fig, width=1000, height=1000)
        # fig.update_layout(scene=dict(aspectmode='data'))
        # fig.show()
        # exit()

    #print("wsx", time.time()-wsx)
    # use furthest point sampling
    if len(np.asarray(pcd.points)) > n_x:
        global ccc,ttt
        ccc+=1
        t_ = time.process_time()
        ids_to_choose = furthest_point_downsample_ids(np.asarray(pcd.points), n_x)
        ttt += time.process_time() - t_
        #print("wsx", time.time()-wsx)
        pcd = pcd.select_by_index(ids_to_choose)
        # pcd = pcd.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([pcd])
        # exit()
    
    x = xn = None
    if len(np.asarray(pcd.points)) > 0:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(10))
        pcd.orient_normals_towards_camera_location()
        # o3d.visualization.draw_geometries([pcd])
        # exit()
    
        x, xn = np.asarray(pcd.points), np.asarray(pcd.normals)

    return x_dense, x, xn
    # return x_plot, x, xn

def compute_sil_idx_at_each_pixel(depth_proc):
    # a silhouette image is the binary image with 1 outside the hand region and 0 inside
    # at each pixel, find index to the closest pixel with value 1, using Distance Transform
    # the points are represented in image frame where origin is at top left; this is consistent with the perspective projection using intrinsic camera parameters
    I_vu_D = distance_transform_edt(~(depth_proc > 0), return_distances=False, return_indices=True)  # (2, H, W) -7 fps
    I_D_vu = np.transpose(I_vu_D, (1, 2, 0)) # (H, W, 2) transpose for ease in array operations later
    # I_D_vu_gray = I_D_vu.astype(np.float32) / mask_sil.shape
    # cv.imshow("dist_trans", np.hstack([I_D_vu_gray[:, :, 0], I_D_vu_gray[:, :, 1]]))

    return I_D_vu

