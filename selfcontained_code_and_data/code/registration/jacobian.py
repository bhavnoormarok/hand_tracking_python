from utils.freq_imports import *




## J_skel
# to compute J_skel (similar to Appendix B in https://lgg.epfl.ch/publications/2015/Htrack_ICP/paper.pdf)
# we need to find the axis (r) and pivot point (a') in world space for all degrees of freedom as per slide 47-56 https://cseweb.ucsd.edu/classes/wi19/cse169-a/slides/CSE169_09.pdf
# a' and r are swapped wrt slides so as to match stretchable bones convention where a' represents pivot point
def compute_axis_per_dof_in_world_space(R_glob,R_glob_init, init_ref_x, init_ref_x_ref_y, Rs_rel, axes):
    
    # # axes in world space
    # axis_per_dof = []
    # # incremental global orientation as euler angles (since the initial global orientation is already close to the groundtruth, it's okay to use euler angles)
    # axis_per_dof.append(R_glob_init @ np.array([1, 0, 0]))
    # axis_per_dof.append(init_ref_x @ np.array([0, 1, 0]))
    # axis_per_dof.append(init_ref_x_ref_y @ np.array([0, 0, 1]))
    # # overall global orientation = inital global orientation @ incremental euler angles
    
    # print(R_glob_init @ np.array([1, 0, 0]))
    # print(R_glob_init[:,0])

    # # R_glob = R_glob_init @ R_glob_ref_x @ R_glob_ref_y @ R_glob_ref_z
    
    # # axes for articulation in world space
    # for i_f in range(5):
    #     i_t_mcp1 = 4*i_f; i_t_mcp2 = i_t_mcp1+1; i_t_pip = i_t_mcp2+1; i_t_dip = i_t_pip+1
    #     # axis for mcp dof 1 will only be affected by global orientation
    #     axis_mcp1 = R_glob @ axes[i_t_mcp1]; axis_per_dof.append(axis_mcp1)
    #     # axis for mcp dof 2 will be affected by it's previous rotations
    #     axis_mcp2 = R_glob @ Rs_rel[i_t_mcp1] @ axes[i_t_mcp2]; axis_per_dof.append(axis_mcp2)
    #     axis_pip = R_glob @ Rs_rel[i_t_mcp1] @ Rs_rel[i_t_mcp2] @ axes[i_t_pip]; axis_per_dof.append(axis_pip)
    #     axis_dip = R_glob @ Rs_rel[i_t_mcp1] @ Rs_rel[i_t_mcp2] @ Rs_rel[i_t_pip] @ axes[i_t_dip]; axis_per_dof.append(axis_dip)
    # axis_per_dof = np.array(axis_per_dof)


    temp1 = R_glob @ Rs_rel[0:20:4]
    temp2 = temp1 @ Rs_rel[1:20:4]
    temp3 = temp2 @ Rs_rel[2:20:4]

    axis_per_dof1 = np.empty((23,3))

    axis_per_dof1[0] = R_glob_init[:,0]
    axis_per_dof1[1] = init_ref_x[:,1]
    axis_per_dof1[2] = init_ref_x_ref_y[:,2]

    axis_per_dof1[3:23:4] = np.einsum("jk,ik->ij", R_glob,axes[0:20:4])
    axis_per_dof1[4:23:4] = np.einsum("ijk,ik->ij",temp1,axes[1:20:4])
    axis_per_dof1[5:23:4] = np.einsum("ijk,ik->ij",temp2,axes[2:20:4])
    axis_per_dof1[6:23:4] = np.einsum("ijk,ik->ij",temp3,axes[3:20:4])

    #print(np.array_equal(axis_per_dof[:3], axis_per_dof1[:3]))

    #exit()
    return axis_per_dof1

def compute_pivot_per_dof_in_world_space(R_glob, t_glob, a_prime):

    print(R_glob)
    print(t_glob)
    print(a_prime)
    exit()


    a_prime_after_glob = a_prime @ R_glob.T + t_glob    # (20, 3)

    # a_primes = []
    # a_primes.append(t_glob)
    # a_primes.append(t_glob)
    # a_primes.append(t_glob)
    # for i_f in range(5):
    #     i_a_mcp = 4*i_f+1; i_a_pip = i_a_mcp+1; i_a_dip = i_a_pip+1
        
    #     # 2 dof at mcp, so repeat the pivot point
    #     a_primes.append(a_prime_after_glob[i_a_mcp])
    #     a_primes.append(a_prime_after_glob[i_a_mcp])
    #     # 1 dof at pip and dip
    #     a_primes.append(a_prime_after_glob[i_a_pip])
    #     a_primes.append(a_prime_after_glob[i_a_dip])
    # a_primes = np.array(a_primes)


    a_primes1 = np.empty((23,3))
    a_primes1[:3] = t_glob
    a_primes1[3:23:4] = a_prime_after_glob[1:20:4]
    a_primes1[4:23:4] = a_primes1[3:23:4]
    a_primes1[5:23:4] = a_prime_after_glob[2:20:4]
    a_primes1[6:23:4] = a_prime_after_glob[3:20:4]

    return a_primes1




def compute_J_skel(y, m_dof_per_y, axis_per_dof, pivot_per_dof):



    # for each point, compute jacobian wrt all rotation dofs
    #qaz = time.time()
    v_pivot_y = y[:, np.newaxis, :] - pivot_per_dof # (|y|, 23, 3)
    #print("qaz1",time.time()-qaz)
    #qaz = time.time()
    
    J_skel_rot = np.cross(axis_per_dof, v_pivot_y, axisc=1)   # (|y|, 23, 3)
    #print("qaz2",time.time()-qaz)
    #az = time.time()

    # for each point, compute jacobian wrt global translation dofs
    # J_skel_trans = np.broadcast_to(np.identity(3)[np.newaxis, :, :], [len(y), 3, 3])    # (|y|, 3, 3)
    # J_skel = np.concatenate([J_skel_trans, J_skel_rot.transpose([0,2,1])], axis=1) # (|y|, 26, 3)
 



    # temp = np.einsum("ijk,ij->ikj", J_skel, m_dof_per_y)
    # J_skel *= m_dof_per_y[:, :, np.newaxis]
    # J_skel = np.transpose(J_skel, [0, 2, 1])    # (|y|, 3, 26)

    J_skel = np.zeros((m_dof_per_y.shape[0],3,26))
    i,j = np.diag_indices(3)
    #print("qaz2.5",time.time()-qaz)
    #qaz = time.time()
    J_skel[:,i,j] = m_dof_per_y[i,j]
    #print("qaz3",time.time()-qaz)
    #qaz = time.time()
    J_skel[:,:,3:] = J_skel_rot * m_dof_per_y[:, None, 3:]
    #print("qaz4",time.time()-qaz)


    return J_skel

def compute_J_persp(x_bg, fx, fy):
    #qaz = time.time()
    J_persp = np.zeros((len(x_bg), 2, 3))   # (|x|, 2, 3)
    J_persp[:, 0, 0] = fx / x_bg[:, 2]
    J_persp[:, 1, 1] = fy / x_bg[:, 2]
    J_persp[:, 0, 2] = -fx * x_bg[:, 0] / (x_bg[:, 2])**2
    J_persp[:, 1, 2] = -fy * x_bg[:, 1] / (x_bg[:, 2])**2
    #print("ewq",time.time()-qaz)
    return J_persp


def compute_J_beta(Rs, R_glob, mano_S, W_bone):
    #print("jagnoor")

    # wer = time.time()
    # n_v, n_b = W_bone.shape

    # # vectorized R
    # #print("S",mano_S.shape)
    # R = jnp.reshape(Rs, (-1, 3))   # (n_b*3, 3)

    # # reshaped mano_S
    # S = jnp.transpose(mano_S, (1, 0, 2)) # (3, n_v, 10)
    # S = jnp.reshape(S, (3, -1))    # (3, n_v*10)

    # # vectorized R*S
    # RS = R @ S # (n_b*3, n_v*10)   # slow step; jax jit helps
    # RS = RS.reshape(n_b, 3, n_v, 10)
    # RS = RS.transpose(2, 1, 3, 0)   # (n_v, 3, 10, n_b)

    # wer = time.time()-wer

    #ert = time.time()
    # RS = np.einsum("ijk,lkn->ljni", Rs, mano_S, optimize="optimal")
    RS = np.tensordot(Rs, mano_S, (2,1))
    #ert = time.time()-ert
    #print(np.allclose(temp,RS), wer, ert)
    
    # vectorized lbs

    RSt = RS.transpose(2, 1, 3, 0)

    
    J_art_beta = np.sum(W_bone[:, np.newaxis, np.newaxis, :] * RSt , axis=3)   # (n_v, 3, 10)  # slow step; jax jit helps
    
    J_beta = np.einsum("ij,kjl->kil",R_glob, J_art_beta)
    # path, display = np.einsum_path("mj,ijkl,ki->kml", R_glob,Rs, mano_S,W_bone, optimize="optimal")
    # path, display = np.einsum_path("mj,ijkl,ki->kml", R_glob,RS,W_bone)
    
    return J_beta


@partial(jit, backend='cpu')
def compute_Jte(J, e):
    Jte = J.T @ e   # (26,)
    return Jte

