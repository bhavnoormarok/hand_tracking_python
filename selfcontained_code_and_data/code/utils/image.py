from utils.freq_imports import *


def mask_to_3c_uint8(mask):
    return (np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255).astype(np.uint8)

def colormap_depth(depth, z_min, z_max, cm=cv.COLORMAP_INFERNO):
    m_bg = (depth < z_min) | (depth > z_max)
    depth_clip = np.clip(depth, z_min, z_max)
    depth_norm = (depth_clip - z_min) / (z_max - z_min)
    depth_cm = cv.applyColorMap((depth_norm*255).astype(np.uint8), cm)[:, :, ::-1]
    depth_cm[m_bg] = 255
    return depth_cm


def calculate_depth_diff_img(depth_proc, depth_ren, diff_threshold):
    m_model_in_front_of_D = depth_proc > (depth_ren+diff_threshold)
    m_model_behind_D = depth_ren > (depth_proc+diff_threshold)
    depth_diff_cm = np.full((depth_proc.shape[0], depth_proc.shape[1], 3), 255, dtype=np.uint8)
    depth_diff_cm[m_model_in_front_of_D] = np.array([255, 0, 0])
    depth_diff_cm[m_model_behind_D] = np.array([0, 0, 255])
    return depth_diff_cm