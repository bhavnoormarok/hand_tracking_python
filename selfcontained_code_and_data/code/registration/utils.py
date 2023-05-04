from utils.freq_imports import *
from utils.array import normalize

def calculate_bone_lengths(k):

        temp = np.copy(k)
        temp[[4,8,12,16,20]] = k[0]
        x = k[1:21] - temp[:20]
        x = np.sqrt(np.einsum("ij,ij->i", x, x))

        return x
        # b = []
        # i_k_wrist = 0
        # for i_f in range(5):
        #     i_k_mcp = 4*i_f + 1; i_k_pip = i_k_mcp+1; i_k_dip = i_k_pip+1; i_k_tip = i_k_dip+1
        #     b_wrist_mcp = np.linalg.norm(k[i_k_mcp] - k[i_k_wrist])
        #     b.append(b_wrist_mcp)

        #     b_mcp_pip = np.linalg.norm(k[i_k_pip] - k[i_k_mcp])
        #     b.append(b_mcp_pip)

        #     b_pip_dip = np.linalg.norm(k[i_k_dip] - k[i_k_pip])
        #     b.append(b_pip_dip)

        #     b_dip_tip = np.linalg.norm(k[i_k_tip] - k[i_k_dip])
        #     b.append(b_dip_tip)

        # b = np.array(b)
        # return b

def sample_face_ids(rng, i_F_per_part, n_f_per_part, n_f, n_s_approx):
    # for each part sample faces proportional to the number of faces belonging to this part
    # qaz = time.time()
    
    # n_s_p = ((n_f_per_part/n_f) * n_s_approx).astype(int)
    # cumsum = np.zeros_like(n_f_per_part)
    # cumsum[1:] = np.cumsum(n_f_per_part)[:-1]
    # sample = np.concatenate([rng.choice(n_f_per_part[i],n_s_p[i])+cumsum[i] for i in range(len(n_s_p))])
    # sample = np.array([0,1,2])
    # i_F_s = i_F_per_part[sample]


    i_F_s = []
    for i_F_p, n_f_p in zip(i_F_per_part, n_f_per_part):
        n_s_p = int((n_f_p/n_f) * n_s_approx) # approximate because of unideal partition
        i_F_s_p = rng.choice(i_F_p, n_s_p)
        i_F_s.extend(i_F_s_p)
    i_F_s = np.array(i_F_s)


    return i_F_s

def generate_random_barycentric_coordinates(rng, n_s):
    """generate random u, v, w for each triangle"""
    # Reference: Section 4.2 eq 1 in https://www.cs.princeton.edu/~funk/tog02.pdf
    r1s = rng.uniform(size=n_s)
    r2s = rng.uniform(size=n_s)

    
    # b_u = 1 - np.sqrt(r1s)
    # b_v = np.sqrt(r1s) * (1 - r2s)
    # b_w = np.sqrt(r1s) * r2s

    temp = np.sqrt(r1s)
    b_u = 1 - temp
    b_w = temp * r2s
    b_v = temp - b_w


    b = np.stack([b_u, b_v, b_w], axis=1)   # (n_s, 3)
    return b

def generate_barycenters_on_mesh(rng, i_F_per_part, n_f_per_part, m_dof_per_face, F, n_s_approx):
    # these barycenters will be evaluated for different mesh pose and then projected onto the image
    # the projected points can then be used for computing the 2d data term

    i_F_s = sample_face_ids(rng, i_F_per_part, n_f_per_part, len(F), n_s_approx=n_s_approx)

    b_s = generate_random_barycentric_coordinates(rng, len(i_F_s))

    m_dof_per_s = m_dof_per_face[i_F_s] # (|b|, 26)

    return i_F_s, b_s, m_dof_per_s

def barycenters_to_mesh_positions(b, i_Fb, v, F):
    pb = b[:, 0:1] * v[F[i_Fb, 0]] + b[:, 1:2] * v[F[i_Fb, 1]] + b[:, 2:3] * v[F[i_Fb, 2]]

    return pb

def barycenters_to_mesh_normals(b, i_Fb, n, F):
    nb = b[:, 0:1] * n[F[i_Fb, 0]] + b[:, 1:2] * n[F[i_Fb, 1]] + b[:, 2:3] * n[F[i_Fb, 2]]
    nb = normalize(nb)
    return nb


