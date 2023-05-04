from utils.freq_imports import *
from utils.helper import create_dir
from utils import plotly_wrapper
from utils.perspective_projection import uvd2xyz, xyz2uv
from utils.moderngl_render import Pointcloud_and_HandRenderer
from utils.image import colormap_depth, calculate_depth_diff_img
from utils.mesh import compute_vertex_normals
from evaluation import metric
from registration import preprocess, global_transform
from registration.pose_registration import PoseRegistration
from registration.shape_registration import ShapeRegistration
from evaluation.kinect.preprocess import process_frame
from multiprocessing import Process, Queue
from registration import preprocess


def get_kinecvtv2_stored_intrinsics():
    fx = 366.085
    fy = 366.085
    cx = 259.229
    cy = 207.968

    return fx, fy, cx, cy




# parser = argparse.ArgumentParser()
# parser.add_argument("user")
# parser.add_argument("sequence", type=int)
# args = parser.parse_args()

shape_reg = ShapeRegistration()
pose_reg = PoseRegistration()





i_k_kinect_palm = pose_reg.i_k_amano_palm
i_k_kinect_reg_k = i_k_amano_reg_k = np.arange(21)
pose_reg.set_i_k_amano_reg_k(i_k_amano_reg_k)
shape_reg.set_i_k_amano_reg_k(i_k_amano_reg_k)

i_k_amano_reinit = np.arange(21)
shape_reg.set_i_k_amano_reinit(i_k_amano_reinit)
pose_reg.set_i_k_amano_reinit(i_k_amano_reinit)

fx, fy, cx, cy = get_kinecvtv2_stored_intrinsics()
shape_reg.set_camera_params(fx, fy, cx, cy)
pose_reg.set_camera_params(fx, fy, cx, cy)

z_near, z_far = 0.6, 1.2   # m
d_near = z_near * 1000; d_far = z_far * 1000    # depth uses mm as units
H, W = 424, 512
point_cloud_and_hand_renderer = Pointcloud_and_HandRenderer(W, H, fx, fy, cx, cy, z_near, z_far, np.array(pose_reg.amano.F), len(pose_reg.amano.v))       

# ----------------------------------------
# NOTE: change this for every user
user = "pratik"
sequence = 4
# user = args.user
# sequence = args.sequence
# ----------------------------------------
path_to_kinect_data = Path(f"./data/kinect")
in_seq_dir = f"{path_to_kinect_data}/{user}/{sequence}"
in_color_raw_dir = f"{in_seq_dir}/color_raw"
in_depth_raw_dir = f"{in_seq_dir}/depth_raw"
in_depth_proc_dir = f"{in_seq_dir}/depth_proc"
color_raw_paths = sorted(Path(in_color_raw_dir).glob("*.npy"))
depth_raw_paths = sorted(Path(in_depth_raw_dir).glob("*.npy"))
depth_proc_paths = sorted(Path(in_depth_proc_dir).glob("*.npy"))

n_frames = len(depth_proc_paths)
print(f"User: {user}, sequence: {sequence}, n_frames = {n_frames}")

out_dir = f"./output/kinect/amano/{user}/{sequence}"; create_dir(out_dir, True)
out_stretch_dir = f"{out_dir}/stretch"; create_dir(out_stretch_dir, False)
out_depth_proc_dir = f"{out_dir}/depth_proc"; create_dir(out_depth_proc_dir, True)
out_depth_ren_dir = f"{out_dir}/depth_ren"; create_dir(out_depth_ren_dir, True)
out_depth_diff_dir = f"{out_dir}/depth_diff"; create_dir(out_depth_diff_dir, True)
out_points_mesh_dir = f"{out_dir}/points_mesh"; create_dir(out_points_mesh_dir, True)
out_metric_dir = f"{out_dir}/metric"; create_dir(out_metric_dir, True)
d2m_mean_file = f"{out_metric_dir}/d2m_mean.txt"; d2m_max_file = f"{out_metric_dir}/d2m_max.txt"
m2d_mean_file = f"{out_metric_dir}/m2d_mean.txt"; m2d_max_file = f"{out_metric_dir}/m2d_max.txt"

metric_cum_avg_file = f"{out_metric_dir}/cum_avg.txt"
with open(metric_cum_avg_file, "w") as file:
    file.write((
        "Frame id"
        " | d2m_mean"
        " | m2d_mean"
        " | d2m_max"
        " | m2d_max"
        "\n"
    ))

cum_d2m_mean = cum_d2m_max = 0.0
cum_m2d_mean = cum_m2d_max = 0.0
cnt_frame = 0

i_frame_start = 0
calibration = True
xyz_crop_center = np.array([0, 0, 0.7])










def writer(qu):
    global point_cloud_and_hand_renderer, out_points_mesh_dir, out_depth_proc_dir, out_depth_ren_dir, out_depth_diff_dir, m2d_max_file, m2d_mean_file, cum_d2m_mean, cum_d2m_max, cum_m2d_max, cum_m2d_mean, cnt_frame
    
    while True:
        x_dense, v_p, n_p, x, xn, y, yn, depth_proc, d_near, d_far, fx, fy, cx, cy, i_frame = qu.get()
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x_dense))
        pcd = pcd.voxel_down_sample(0.003)
        x_plot = np.asarray(pcd.points)
        point_cloud_and_hand_renderer.write_vbo(v_p, n_p, x_plot, x, xn, y, yn)
        point_cloud_and_hand_renderer.render_x_dense_and_mesh()
        color = point_cloud_and_hand_renderer.extract_fbo_color()
        cv.imwrite(f"{out_points_mesh_dir}/{i_frame:05d}.png", color[:, :, ::-1])

        # input processed depth
        depth_proc_cm = colormap_depth(depth_proc, d_near, d_far, cv.COLORMAP_INFERNO)
        cv.imwrite(f"{out_depth_proc_dir}/{i_frame:05d}.png", depth_proc_cm[:, :, ::-1])

        # rendered depth
        point_cloud_and_hand_renderer.render_model()
        depth_ren = point_cloud_and_hand_renderer.extract_fbo_depth()
        depth_ren_cm = colormap_depth(depth_ren, d_near, d_far, cv.COLORMAP_INFERNO)
        cv.imwrite(f"{out_depth_ren_dir}/{i_frame:05d}.png", depth_ren_cm[:, :, ::-1])
        
        # depth difference
        depth_diff_cm = calculate_depth_diff_img(depth_proc, depth_ren, diff_threshold=10)
        cv.imwrite(f"{out_depth_diff_dir}/{i_frame:05d}.png", depth_diff_cm[:, :, ::-1])


        ## write metric
        d2m_mean, d2m_max = metric.compute_d2m_mean_max(depth_proc, depth_ren, fx, fy, cx, cy)
        with open(d2m_mean_file, "a") as file:
            file.write(f"{d2m_mean:.3f}\n")
        with open(d2m_max_file, "a") as file:
            file.write(f"{d2m_max:.3f}\n")
        
        m2d_mean, m2d_max = metric.compute_m2d_mean_max(depth_proc, depth_ren)
        with open(m2d_mean_file, "a") as file:
            file.write(f"{m2d_mean:.3f}\n")
        with open(m2d_max_file, "a") as file:
            file.write(f"{m2d_max:.3f}\n")

        cum_d2m_mean += d2m_mean; cum_d2m_max += d2m_max
        cum_m2d_mean += m2d_mean; cum_m2d_max += m2d_max
        cnt_frame += 1
        metric_cum_avg_str = (
            f"{i_frame:05d}"
            f"| {cum_d2m_mean/cnt_frame:.3f}"
            f"| {cum_m2d_mean/cnt_frame:.3f}"
            f"| {cum_d2m_max/cnt_frame:.3f}"
            f"| {cum_m2d_max/cnt_frame:.3f}"
            "\n"
        )
        with open(metric_cum_avg_file, 'a') as file:
            file.write(metric_cum_avg_str)
        






if __name__ == '__main__':


    # qu = Queue()
    # writer_process = Process(target = writer, args=(qu,))
    # writer_process.start()

    process2 = Process(target = pose_reg.thread2, args=())
    process2.start()
    process3 = Process(target = pose_reg.thread3, args=())
    process3.start()


    for i_frame in tqdm(range(n_frames), dynamic_ncols=True):
        depth_proc = np.load(f"{depth_proc_paths[i_frame]}")
        # color_raw = cv.imread(f"{color_raw_paths[i_frame]}")[:, :, ::-1]
        # depth_raw = np.load(f"{depth_raw_paths[i_frame]}")
        # color_proc, depth_proc, mask_sil, xyz_crop_center = process_frame(color_raw, depth_raw, d_near, d_far, fx, fy, cx, cy, xyz_crop_center)
        qaz = time.time()
        I_D_vu = preprocess.compute_sil_idx_at_each_pixel(depth_proc)
        #print("qaz",time.time()-qaz)
        qaz = time.time()
        x_dense, x, xn = preprocess.depth_to_point_cloud(depth_proc, fx, fy, cx, cy, n_x=pose_reg.n_x)
        
        if x is None:
            print(f"Frame {i_frame}, invalid point cloud")
            continue

        if i_frame == i_frame_start:
            ## use marked keypoints to initialize params of first frame
            
            # read marked keypoints for user
            k_marked = np.load(f"output/kinect/marked_keypoints/{user}/{sequence}/k_marked.npy").astype(np.float32)

            # init params
            beta = np.zeros(10)
            phi, k_s = pose_reg.calculate_phi(k_marked, beta, return_k_s=True)
            R_glob_init, t_glob = global_transform.compute_global_trans_from_palm_keypoints(k_marked[i_k_kinect_palm], k_s[pose_reg.i_k_amano_palm])
            theta_glob = np.zeros(3)
            theta = np.zeros(20)
            k_p_prev = None

            # register pose to keypoints
            v_p, n_p, k_p, axis_per_dof, pivot_per_dof = pose_reg.deform_and_compute_linearized_info(phi, beta, R_glob_init, theta_glob, t_glob, theta)
            theta_glob, t_glob, theta, v_p, n_p, k_p, axis_per_dof, pivot_per_dof = pose_reg.register_to_keypoints(
                k_marked[i_k_kinect_reg_k], 
                phi, beta, R_glob_init, theta_glob, t_glob, theta,
                v_p, n_p, k_p, axis_per_dof, pivot_per_dof,
                k_p_prev,
            )

            # add residual global rotation into initial global rotation
            R_glob_ref_x = Rotation.from_euler('X', theta_glob[0]).as_matrix()
            R_glob_ref_y = Rotation.from_euler('Y', theta_glob[1]).as_matrix()
            R_glob_ref_z = Rotation.from_euler('Z', theta_glob[2]).as_matrix()
            R_glob = R_glob_init @ R_glob_ref_x @ R_glob_ref_y @ R_glob_ref_z
            R_glob_init = R_glob.copy() # use this as initial global transformation for next frame
            theta_glob = np.zeros(3)

            # ready for first iteration of shape registration
            v_p, n_p, k_p, axis_per_dof, pivot_per_dof, J_beta = shape_reg.deform_and_compute_linearized_info(phi, beta, R_glob_init, theta_glob, t_glob, theta)

            k_data = k_marked
        else:
            # use previous frame's global transformation to initialize current frame's global transformation
            R_glob_ref_x = Rotation.from_euler('X', theta_glob[0]).as_matrix()
            R_glob_ref_y = Rotation.from_euler('Y', theta_glob[1]).as_matrix()
            R_glob_ref_z = Rotation.from_euler('Z', theta_glob[2]).as_matrix()
            R_glob = R_glob_init @ R_glob_ref_x @ R_glob_ref_y @ R_glob_ref_z
            R_glob_init = R_glob.copy() # use this as initial global transformation for next frame

            # since theta_glob captures offset from initial global transform, init to zero
            theta_glob = np.zeros(3)
            # t_glob, theta, beta are initialized using previous frame's estimates

            k_p_prev = k_p
            # there are no keypoints for other frames
            k_data = None

        # register to point cloud
        if calibration:
            beta, theta_glob, t_glob, theta, v_p, n_p, k_p, axis_per_dof, pivot_per_dof, J_beta, y, yn, p, q = shape_reg.register_to_pointcloud(
                k_data, x, xn, I_D_vu,
                phi, beta, R_glob_init, theta_glob, t_glob, theta,
                v_p, n_p, k_p, axis_per_dof, pivot_per_dof, J_beta,
                k_p_prev
            )

            if (i_frame - i_frame_start) == 50:
                print("calibration complete")
                calibration = False
                np.save(f"{out_dir}/beta.npy", beta)

                ## plot template, shaped and stretched mesh
                
                # rotate and translate for rendering
                R_rot_temp_to_face_cam = Rotation.from_euler("XYZ", [90, 90, 0], degrees=True).as_matrix()
                t_vis = np.array([0, 0, 0.7])

                # template
                v_vis = shape_reg.amano.v @ R_rot_temp_to_face_cam.T + t_vis
                n_vis = compute_vertex_normals(v_vis, shape_reg.amano.F)
                point_cloud_and_hand_renderer.write_vbo_model(v_vis, n_vis)
                point_cloud_and_hand_renderer.render_model()
                color = point_cloud_and_hand_renderer.extract_fbo_color()
                cv.imwrite(f"{out_stretch_dir}/template_color.png", color[:, :, ::-1])
                depth_ren = point_cloud_and_hand_renderer.extract_fbo_depth()
                depth_ren_cm = colormap_depth(depth_ren, d_near, d_far, cv.COLORMAP_INFERNO)
                cv.imwrite(f"{out_stretch_dir}/template_depth.png", depth_ren_cm[:, :, ::-1])
                
                # stretched
                v_stretch = shape_reg.amano.deform(phi, np.zeros(10), np.zeros(20))
                v_vis = v_stretch @ R_rot_temp_to_face_cam.T + t_vis
                n_vis = compute_vertex_normals(v_vis, shape_reg.amano.F)
                point_cloud_and_hand_renderer.write_vbo_model(v_vis, n_vis)
                point_cloud_and_hand_renderer.render_model()
                color = point_cloud_and_hand_renderer.extract_fbo_color()
                cv.imwrite(f"{out_stretch_dir}/stretched_color.png", color[:, :, ::-1])  
                depth_ren = point_cloud_and_hand_renderer.extract_fbo_depth()
                depth_ren_cm = colormap_depth(depth_ren, d_near, d_far, cv.COLORMAP_INFERNO)
                cv.imwrite(f"{out_stretch_dir}/stretched_depth.png", depth_ren_cm[:, :, ::-1])

                # shaped
                v_shaped = shape_reg.amano.deform(np.ones(20), beta, np.zeros(20))
                v_vis = v_shaped @ R_rot_temp_to_face_cam.T + t_vis
                n_vis = compute_vertex_normals(v_vis, shape_reg.amano.F)
                point_cloud_and_hand_renderer.write_vbo_model(v_vis, n_vis)
                point_cloud_and_hand_renderer.render_model()
                color = point_cloud_and_hand_renderer.extract_fbo_color()
                cv.imwrite(f"{out_stretch_dir}/shaped_color.png", color[:, :, ::-1])  
                depth_ren = point_cloud_and_hand_renderer.extract_fbo_depth()
                depth_ren_cm = colormap_depth(depth_ren, d_near, d_far, cv.COLORMAP_INFERNO)
                cv.imwrite(f"{out_stretch_dir}/shaped_depth.png", depth_ren_cm[:, :, ::-1])

                # stretched and shaped
                v_stretch_shaped = shape_reg.amano.deform(phi, beta, np.zeros(20))
                v_vis = v_stretch_shaped @ R_rot_temp_to_face_cam.T + t_vis
                n_vis = compute_vertex_normals(v_vis, shape_reg.amano.F)
                point_cloud_and_hand_renderer.write_vbo_model(v_vis, n_vis)
                point_cloud_and_hand_renderer.render_model()
                color = point_cloud_and_hand_renderer.extract_fbo_color()
                cv.imwrite(f"{out_stretch_dir}/stretched_shaped_color.png", color[:, :, ::-1])  
                depth_ren = point_cloud_and_hand_renderer.extract_fbo_depth()
                depth_ren_cm = colormap_depth(depth_ren, d_near, d_far, cv.COLORMAP_INFERNO)
                cv.imwrite(f"{out_stretch_dir}/stretched_shaped_depth.png", depth_ren_cm[:, :, ::-1])
            
        else:
            theta_glob, t_glob, theta, v_p, n_p, k_p, axis_per_dof, pivot_per_dof, y, yn, p, q = pose_reg.register_to_pointcloud(
                k_data, x, xn, I_D_vu,
                phi, beta, R_glob_init, theta_glob, t_glob, theta,
                v_p, n_p, k_p, axis_per_dof, pivot_per_dof,
                k_p_prev,
            )

        ## plot

        x_dense_old = x_dense.copy()
        v_p_old = v_p.copy()
        n_p_old = n_p
        x_old = x.copy()
        xn_old = xn.copy()
        y_old = y.copy()
        yn_old = yn.copy()
        depth_proc_old = depth_proc.copy()
        d_near_old = d_near
        d_far_old = d_far
        fx_old = fx
        fy_old = fy
        cx_old = cx
        cy_old = cy

        
        #qu.put((x_dense_old, v_p_old, n_p_old, x_old, xn_old, y_old, yn_old, depth_proc_old, d_near_old, d_far_old, fx_old, fy_old, cx_old, cy_old, i_frame))
        
        continue
        # mesh and point cloud; downsample point cloud so that it's not too dense
        
    #writer_process.join()



    subprocess.run(["./code/utils/create_video_from_frames.sh", "-f", "30", "-s", f"{i_frame_start}", "-w", "%05d.png", f"{out_depth_proc_dir}"])
    subprocess.run(["./code/utils/create_video_from_frames.sh", "-f", "30", "-s", f"{i_frame_start}", "-w", "%05d.png", f"{out_depth_ren_dir}"])
    subprocess.run(["./code/utils/create_video_from_frames.sh", "-f", "30", "-s", f"{i_frame_start}", "-w", "%05d.png", f"{out_depth_diff_dir}"])
    subprocess.run(["./code/utils/create_video_from_frames.sh", "-f", "30", "-s", f"{i_frame_start}", "-w", "%05d.png", f"{out_points_mesh_dir}"])

    print(preprocess.ccc,preprocess.ttt)