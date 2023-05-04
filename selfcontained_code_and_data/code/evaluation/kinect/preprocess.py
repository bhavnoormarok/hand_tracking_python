from utils.freq_imports import *
from utils.perspective_projection import uvd2xyz, xyz2uv
from utils.array import normalize
from utils.image import colormap_depth, mask_to_3c_uint8


def compute_wristband_mask(color_hsv, min_hsv, max_hsv):
    # mask_wb_h = (color_hsv[:, :, 0] >= min_hsv[0]) & (color_hsv[:, :, 0] <= max_hsv[0])
    # mask_wb_s = (color_hsv[:, :, 1] >= min_hsv[1]) & (color_hsv[:, :, 1] <= max_hsv[1])
    # mask_wb_v = (color_hsv[:, :, 2] >= min_hsv[2]) & (color_hsv[:, :, 2] <= max_hsv[2])
    # mask_wb_hsv = mask_wb_h & mask_wb_s & mask_wb_v
    # return False, np.zeros_like(color_hsv[:, :, 0])
    mask_wb_hsv = cv.inRange(color_hsv, min_hsv, max_hsv).astype(bool)
    # mask_wb_hsv = np.all((color_hsv >= min_hsv) & (color_hsv <= max_hsv), axis=2)
    # use connected components to identify wristband region as the 2nd most maximum area (1st max is background)
    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask_wb_hsv.astype(np.uint8))
    # labels.shape = (H, W)
    # stats.shape = (n_labels, 5) Ref: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
    # centroids.shape = (n_labels, 2)
    if n_labels < 2:    # no component except background
        return False, np.zeros_like(mask_wb_hsv)
    label_ids_sorted_by_area = np.argsort(stats[:, 4])  # 4 represents area of each component
    label_wb_comp = np.arange(n_labels)[label_ids_sorted_by_area][-2]
    mask_wb = labels == label_wb_comp

    # cm_label = plt.get_cmap('inferno')
    # labels_cm = (cm_label(labels/labels.max())[:, :, :3] * 255).astype(np.uint8) 
    # cv.imshow("label", np.hstack([color_hsv[:, :, ::-1], labels_cm, mask_to_3c_uint8(mask_wb)]))

    return True, mask_wb  



def process_frame(color_raw, depth_raw, d_near, d_far, fx, fy, cx, cy, xyz_crop_center_prev):
    # depth_raw_cm = colormap_depth(depth_raw, 10, 2000, cv.COLORMAP_HOT)
    # cv.imshow("raw", np.hstack([color_raw[:, :, ::-1], depth_raw_cm[:, :, ::-1]]))

    # clip beyond far
    color_proc = color_raw.copy(); depth_proc = depth_raw.copy()
    mask_near_far = (depth_raw > d_near) & (depth_raw < d_far)
    depth_proc[~mask_near_far] = 0
    # color_proc[~mask_near_far] = 0    # this is very slow (possible issue with broadcasting)
    color_proc[~np.repeat(mask_near_far[:, :, np.newaxis], 3, axis=2)] = 0  # this is fast (manually repeating is faster)
    # depth_proc_cm = colormap_depth(depth_proc, d_near, d_far, cv.COLORMAP_HOT)
    # cv.imshow("clip", np.hstack([color_raw[:, :, ::-1], depth_raw_cm[:, :, ::-1], color_proc[:, :, ::-1], depth_proc_cm[:, :, ::-1], mask_to_3c_uint8(mask_near_far)]))

    # Algorithm to crop hand region using color and depth
    # 1. identify wristband mask using color image
    # 2. compute its centroid
    # 3. remove pixels that are not within a depth range of the wrist 
    # 4. principal direction of variation of wristband (PCA)
    # 5. offset the centroid along principal direction and use it as a center of a sphere to crop
    
    # 1. segment wristband in hsv
    color_proc_hsv = cv.cvtColor(color_proc, cv.COLOR_RGB2HSV)
    color_proc_hsv_blur = cv.medianBlur(color_proc_hsv, 5)    # -5 fps
    hsv_bounds_green = np.array([
        [50, 120],
        [100, 255],
        [0, 255],
    ])
    found_wb, mask_wb = compute_wristband_mask(color_proc_hsv_blur, hsv_bounds_green[:, 0], hsv_bounds_green[:, 1])  # -5 fps
    # dilate to remove boundary
    mask_wb = cv.dilate(mask_wb.astype(np.uint8), np.ones((5,5),np.uint8)).astype(bool)
    # cv.imshow("hsv", np.hstack([color_proc[:, :, ::-1], color_proc_hsv_blur[:, :, ::-1], mask_to_3c_uint8(mask_wb)]))    # -8 fps due to visualizing mask with mask_to_3c_uint8

    # back-project each pixel coordinate
    h, w = depth_proc.shape
    V, U = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    X = (U - cx) * depth_proc / fx
    Y = (V - cy) * depth_proc / fy
    Z = depth_proc

    crop_radius = 200; wb_size = -10
    found_crop_center = False
    if found_wb:
        # 2. compute wristband centroid    -5 fps
        vu_wb = np.argwhere(mask_wb)
        uvd_wb = np.stack([vu_wb[:, 1], vu_wb[:, 0], depth_proc[mask_wb]], axis=1)
        xyz_wb = uvd2xyz(uvd_wb, fx, fy, cx, cy)
        xyz_avg_wb = np.mean(xyz_wb, axis=0)
        uv_avg_wb = xyz2uv(xyz_avg_wb, fx, fy, cx, cy)
        color_proc = cv.drawMarker(color_proc, uv_avg_wb.astype(int), (0, 255, 0))

        # 3. remove pixels that are not within a depth range of the wrist
        # depth_range = 150
        # mask_wrist_range = (depth_proc > (xyz_avg_wb[2]-depth_range)) & (depth_proc < (xyz_avg_wb[2]+depth_range))
        dist_sq_from_wb = (X - xyz_avg_wb[0])**2 + (Y - xyz_avg_wb[1])**2 + (Z - xyz_avg_wb[2])**2
        wrist_range_radius = 200
        mask_wrist_range = dist_sq_from_wb < (wrist_range_radius*wrist_range_radius)
        # cv.imshow("wrist_range", np.hstack([color_proc[:, :, ::-1], depth_proc_cm[:, :, ::-1], mask_to_3c_uint8(mask_wrist_range)]))


        # 4. compute direction of variation of points near the wristband centroid
        # compute points inside the silhouette so far
        if np.count_nonzero(mask_wrist_range) > 1000:

            vu_wrist_range = np.argwhere(mask_wrist_range)
            uvd_wrist_range = np.stack([vu_wrist_range[:, 1], vu_wrist_range[:, 0], depth_proc[mask_wrist_range]], axis=1)
            xyz_wrist_range = uvd2xyz(uvd_wrist_range, fx, fy, cx, cy)
            
            # compute PCA (Ref: https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python)
            xyz_wrist_range_centered = xyz_wrist_range - np.mean(xyz_wrist_range, axis=0)
            cov = np.cov(xyz_wrist_range_centered, rowvar=False)
            evals, evecs = scipy.linalg.eigh(cov)  # use eigh instead of eig since our matrix is symmetric; performance gain is substantial
            # sort in decreasing order of eigenvalue
            ids = np.argsort(evals)[::-1]
            evals = evals[ids]; evecs = evecs[:, ids]
            dir_max_var = normalize(evecs[:, 0])
            
            # Assumption: arm is always below the wrist (i.e. it is along positive Y axis wrt camera frame)
            # allow wrist to point downward
            if dir_max_var[1] > 0:
                dir_max_var = -dir_max_var
            
            xyz_crop_center = xyz_avg_wb + dir_max_var * (crop_radius + wb_size)
            found_crop_center = True
    
    if not found_crop_center:
        xyz_crop_center = xyz_crop_center_prev

    # uv_crop_center = xyz2uv(xyz_crop_center, fx, fy, cx, cy)
    # color_proc = cv.drawMarker(color_proc, uv_crop_center.astype(int), (0, 255, 0))
    # cv.imshow("crop_center", np.hstack([color_proc[:, :, ::-1]]))

    dist_sq_from_crop_center = (X - xyz_crop_center[0])**2 + (Y - xyz_crop_center[1])**2 + (Z - xyz_crop_center[2])**2
    mask_sphere = dist_sq_from_crop_center < (crop_radius*crop_radius)
    
    mask_sil = mask_sphere & ~mask_wb
    
    depth_proc[~mask_sil] = 0
    color_proc[~np.repeat(mask_sil[:, :, np.newaxis], 3, axis=2)] = 0
    uv_crop_center = xyz2uv(xyz_crop_center, fx, fy, cx, cy)
    color_proc = cv.drawMarker(color_proc, uv_crop_center.astype(int), (0, 255, 0))
    # depth_raw_cm = colormap_depth(depth_raw, d_near, d_far, cv.COLORMAP_HOT)
    # depth_proc_cm = colormap_depth(depth_proc, d_near, d_far, cv.COLORMAP_HOT)
    # cv.imshow("sil", np.hstack([color_raw[:, :, ::-1], depth_raw_cm[:, :, ::-1], color_proc[:, :, ::-1], depth_proc_cm[:, :, ::-1]]))
    # cv.imshow("crop", np.hstack([color_proc[:, :, ::-1], depth_proc_cm[:, :, ::-1]]))
    
    return color_proc, depth_proc, mask_sil, xyz_crop_center


