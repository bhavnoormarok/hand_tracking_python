from utils.freq_imports import *
from hand_model.amano import Amano
from registration.base import sphere_proxy, dof_mask, part_face
from registration import utils

class RegistrationBase:
    def __init__(self):
        self.amano = Amano()

        # for angle limit
        theta_bounds_dir = './output/hand_model/prior/bounds'
        self.theta_min = np.load(f'{theta_bounds_dir}/theta_min.npy')
        self.theta_max = np.load(f'{theta_bounds_dir}/theta_max.npy')

        # for pca pose prior
        pca_dir = './output/hand_model/prior/pca'
        self.mu = np.load(f'{pca_dir}/mu.npy')   # (20,)
        self.Pi = np.load(f'{pca_dir}/Pi.npy')   # (20, 20)
        self.Sigma = np.load(f'{pca_dir}/Sigma.npy') # (20, 20)

        # for intersection penalty
        self.i_s_per_pair, self.r_per_sphere, self.i_v_per_sphere = sphere_proxy.get_proxy_details()

        # 3d correspondences
        self.m_dof_per_vert = dof_mask.compute_dof_mask_per_vert(self.amano.W_bone)
        self.m_dof_per_face = dof_mask.compute_dof_mask_per_face(self.m_dof_per_vert, self.amano.F)
        self.i_F_per_part, self.n_f_per_part = part_face.compute_face_ids_per_part(self.amano.F, self.amano.W_bone)

        # for 2d correspondences
        n_s_bg_approx = 500
        self.rng = np.random.default_rng(1)
        self.i_F_bg, self.b_bg, self.m_dof_per_bg = utils.generate_barycenters_on_mesh(self.rng, self.i_F_per_part, self.n_f_per_part, self.m_dof_per_face, self.amano.F, n_s_bg_approx)

        # for keypoint data term
        I_v_surr_k = np.load(f'./output/hand_model/vertex_ids_surr_keypoints.npy')

        self.m_dof_per_k = np.any(self.m_dof_per_vert[I_v_surr_k], axis=1)   # (21, 26)
        
        # global transform
        self.i_k_amano_palm = np.array(
            [0, 1, 5, 9, 13, 17]
        )
        
    def set_i_k_amano_reinit(self, i_k_amano_reinit):
        self.i_k_amano_reinit = i_k_amano_reinit

    def set_i_k_amano_reg_k(self, i_k_amano_reg_k):
        self.i_k_amano_reg_k = i_k_amano_reg_k

    def set_camera_params(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def calculate_phi(self, k_data, beta, return_k_s=False):
        # shape blends

        v_s = self.amano.v + self.amano.mano_S @ beta  # (|v|, 3)
        # obtain keypoint positions in shaped mesh
        k_s = self.amano.K @ v_s  # (21, 3)


        b_amano = utils.calculate_bone_lengths(k_s)

        b_data = utils.calculate_bone_lengths(k_data)

        phi = b_data / b_amano

        if return_k_s:
            return phi, k_s

        return phi

    