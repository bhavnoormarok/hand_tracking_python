from utils.freq_imports import *
from registration import utils
from utils.perspective_projection import uvd2xyz, xyz2uv


@partial(jit, backend='cpu')
def compute_closest_point_on_mesh(x, xn, p_s, n_s, w_pos, w_nor):
    # for each data point, compute distances wrt all model points
    D_x_p_pos = np.sum((x[:, np.newaxis, :] - p_s)**2, axis=2)
    D_x_p_nor = np.sum((xn[:, np.newaxis, :] - n_s)**2, axis=2)
    D_x_p = w_pos * D_x_p_pos + w_nor * D_x_p_nor    # (|x|, |p_s|)

    # corresponding model point is the point with minimum distance
    i_p_s_per_x = np.argmin(D_x_p, axis=1)  # (|x|,)
    d_x_p = D_x_p[np.arange(len(x)), i_p_s_per_x]   # (|x|,)

    return i_p_s_per_x, d_x_p

def compute_3d_correspondences(v_p, n_p, x, xn, rng, F, i_F_per_part, n_f_per_part, m_dof_per_face, w_pos, w_nor, n_s_approx):
    # sample barycenters
    #wsx = time.time()
    i_F_s = utils.sample_face_ids(rng, i_F_per_part, n_f_per_part, len(F), n_s_approx=n_s_approx)    # 212 results in 200 points
    #print("wsxy",time.time()-wsx)
    #wsx = time.time()

    b_s = utils.generate_random_barycentric_coordinates(rng, len(i_F_s))
    #print("wsx",time.time()-wsx)
    #wsx = time.time()
    m_dof_per_s = m_dof_per_face[i_F_s] # (|b|, 26)
    # evaluate barycenters on given mesh
    #print("wsx",time.time()-wsx)
    #wsx = time.time()
    p_s = utils.barycenters_to_mesh_positions(b_s, i_F_s, v_p, F)
    #print("wsx",time.time()-wsx)
    #wsx = time.time()
    n_s = utils.barycenters_to_mesh_normals(b_s, i_F_s, n_p, F)
    # for each x, compute closest points from the evaluated points
    #print("wsx",time.time()-wsx)
    wsx = time.time()
    
    i_p_s_per_x, d_x_p = compute_closest_point_on_mesh(x, xn, p_s, n_s, w_pos, w_nor)
    #print("qwe",time.time()-wsx)
    #wsx = time.time()
    # p_scat = plotly_utils.scatter3d(p_s, size=5)
    # n_cone = plotly_utils.cone3d(p_s, n_s, size=10, opacity=1, colorscale='Blackbody')
    # c = np.zeros(len(F)); c[i_F_s] = 1
    # mesh = plotly_utils.mesh3d(v, F, intensitymode='cell', intensity=c, opacity=0.5)
    # fig = go.Figure([mesh, p_scat, n_cone])
    # plotly_utils.remove_fig_background(fig)
    # plotly_utils.update_fig_size(fig)
    # fig.show()

    y = p_s[i_p_s_per_x]    # (|x|, 3)
    yn = n_s[i_p_s_per_x]   # (|x|, 3)
    i_F_y = i_F_s[i_p_s_per_x]  # (|x|,)
    b_y = b_s[i_p_s_per_x]  # (|x|, 3)
    m_dof_per_y = m_dof_per_s[i_p_s_per_x]  # (|x|, 26)

    return y, yn, i_F_y, b_y, m_dof_per_y

def update_3d_correspondences(v_p, n_p, i_F_y, b_y, m_dof_per_y, x, xn, rng, F, i_F_per_part, n_f_per_part, m_dof_per_face, w_pos, w_nor, n_s_approx):
    # evaluate position for previous barycenters for new pose
    y = utils.barycenters_to_mesh_positions(b_y, i_F_y, v_p, F)
    yn = utils.barycenters_to_mesh_normals(b_y, i_F_y, n_p, F)
    d_x_y = w_pos * np.sum((x - y)**2, axis=1) + w_nor * np.sum((xn - yn)**2, axis=1)

    # sample new barycenters for this iteration
    i_F_s = utils.sample_face_ids(rng, i_F_per_part, n_f_per_part, len(F), n_s_approx=n_s_approx)
    b_s = utils.generate_random_barycentric_coordinates(rng, len(i_F_s))
    m_dof_per_s = m_dof_per_face[i_F_s] # (|b|, 26)
    # evaluate new barycenters on new pose
    p_s = utils.barycenters_to_mesh_positions(b_s, i_F_s, v_p, F)
    n_s = utils.barycenters_to_mesh_normals(b_s, i_F_s, n_p, F)
    # for each x, compute closest points from the newly evaluated points
    i_p_s_per_x, d_x_p = compute_closest_point_on_mesh(x, xn, p_s, n_s, w_pos, w_nor)
    i_p_s_per_x = np.array(i_p_s_per_x); d_x_p = np.array(d_x_p)

    # for each x, update the closest point if any of the new points are closer than previous points
    m_p_over_y = d_x_p < d_x_y
    y_new = np.where(m_p_over_y[:, np.newaxis], p_s[i_p_s_per_x], y)
    yn_new = np.where(m_p_over_y[:, np.newaxis], n_s[i_p_s_per_x], yn)
    i_F_y_new = np.where(m_p_over_y, i_F_s[i_p_s_per_x], i_F_y)
    b_y_new = np.where(m_p_over_y[:, np.newaxis], b_s[i_p_s_per_x], b_y)
    m_dof_per_y_new = np.where(m_p_over_y[:, np.newaxis], m_dof_per_s[i_p_s_per_x], m_dof_per_y)

    return y_new, yn_new, i_F_y_new, b_y_new, m_dof_per_y_new

def compute_2d_correspondences(v_p, I_D_vu, F, i_F_bg, b_bg, fx, fy, cx, cy):
    # evaluate barycenters on given mesh
    x_bg = utils.barycenters_to_mesh_positions(b_bg, i_F_bg, v_p, F)  # (|b_bg|, 3)
    p = xyz2uv(x_bg, fx, fy, cx, cy).astype(int)   # (|b_bg|, 2)
    
    # compute closest point on silhouette for each p
    p[:, 0] = np.clip(p[:, 0], 0, I_D_vu.shape[1]-1); p[:, 1] = np.clip(p[:, 1], 0, I_D_vu.shape[0]-1)  # avoid point to project outside image bounds
    q_vu = I_D_vu[p[:, 1], p[:, 0]] # (|b_bg|, 2)
    q = np.fliplr(q_vu)

    return x_bg, p, q