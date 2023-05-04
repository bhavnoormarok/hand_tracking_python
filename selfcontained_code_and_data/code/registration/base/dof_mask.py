import numpy as np
import time
## sample faceids per part, also compute dof influence for each face
def compute_dof_mask_per_vert(W_bone):
    # for each vertex, identify dofs that it influences

    m_bone_per_vert = W_bone > 0.1   # (|v|, 20)
    m_dof_per_vert = np.full((len(W_bone), 26), False)  # (|v|, 26)
    for i_f in range(5):
        wrist_id = 4*i_f; mcp_id = wrist_id+1; pip_id = mcp_id+1; dip_id = pip_id+1
        i_t_mcp1 = 4*i_f; i_t_mcp2 = i_t_mcp1+1; i_t_pip = i_t_mcp2+1; i_t_dip = i_t_pip+1

        m_dof_per_vert[:, :6] |= (m_bone_per_vert[:, wrist_id] | m_bone_per_vert[:, mcp_id] | m_bone_per_vert[:, pip_id] | m_bone_per_vert[:, dip_id])[:, np.newaxis]
        m_dof_per_vert[:, 6+i_t_mcp1] = m_bone_per_vert[:, mcp_id] | m_bone_per_vert[:, pip_id] | m_bone_per_vert[:, dip_id]
        m_dof_per_vert[:, 6+i_t_mcp2] = m_bone_per_vert[:, mcp_id] | m_bone_per_vert[:, pip_id] | m_bone_per_vert[:, dip_id]
        m_dof_per_vert[:, 6+i_t_pip] = m_bone_per_vert[:, pip_id] | m_bone_per_vert[:, dip_id]
        m_dof_per_vert[:, 6+i_t_dip] = m_bone_per_vert[:, dip_id]

        # m_dof_per_vert[:, :6] |= (m_bone_per_vert[:, wrist_id])[:, np.newaxis]
        # m_dof_per_vert[:, 6+i_t_mcp1] = m_bone_per_vert[:, mcp_id] 
        # m_dof_per_vert[:, 6+i_t_mcp2] = m_bone_per_vert[:, mcp_id]
        # m_dof_per_vert[:, 6+i_t_pip] = m_bone_per_vert[:, pip_id]
        # m_dof_per_vert[:, 6+i_t_dip] = m_bone_per_vert[:, dip_id]


    return m_dof_per_vert

def compute_dof_mask_per_face(m_dof_per_vert, F):
    # for each face, identify dofs that it influences
    m_dof_per_face = m_dof_per_vert[F[:, 0]] | m_dof_per_vert[F[:, 1]] | m_dof_per_vert[F[:, 2]]    # (|F|, 26)
    return m_dof_per_face
