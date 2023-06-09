import numpy as np
import time
def get_proxy_details():
    # Note: Sphere are defined from tip to MCP
    '''
    n_spheres_per_finger = [4, 6, 7, 6, 5]
    n_spheres_cumulative = np.cumsum(n_spheres_per_finger)

    sphere_ids_per_fingers = [
        list(range(0, n_spheres_per_finger[0])),
        list(range(n_spheres_cumulative[0], n_spheres_cumulative[1])),
        list(range(n_spheres_cumulative[1], n_spheres_cumulative[2])),
        list(range(n_spheres_cumulative[2], n_spheres_cumulative[3])),
        list(range(n_spheres_cumulative[3], n_spheres_cumulative[4])),
    ]

    # sphere pairs
    sphere_id_pairs = []
    for finger_id_1 in range(5):
        for sphere_id_1 in sphere_ids_per_fingers[finger_id_1]:
            for finger_id_2 in range(finger_id_1+1, 5):
                for id_sphere_id_2, sphere_id_2 in enumerate(sphere_ids_per_fingers[finger_id_2]):

                    # dont include neighbor finger's base(last) sphere, prevents neigbor finger vertices from influencing current finger
                    n_spheres_in_finger_2 = len(sphere_ids_per_fingers[finger_id_2])
                    if id_sphere_id_2 == n_spheres_in_finger_2 - 1:
                        continue
                    
                    sphere_id_pairs.append([sphere_id_1, sphere_id_2])
    sphere_id_pairs = np.array(sphere_id_pairs) # (257, 2)
    '''
    
    sphere_id_pairs = np.array([[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],[0,14],[0,15],[0,17],[0,18],[0,19],[0,20],[0,21],[0,23],[0,24],[0,25],[0,26],[1,4],[1,5],[1,6],[1,7],[1,8],[1,10],[1,11],[1,12],[1,13],[1,14],[1,15],[1,17],[1,18],[1,19],[1,20],[1,21],[1,23],[1,24],[1,25],[1,26],[2,4],[2,5],[2,6],[2,7],[2,8],[2,10],[2,11],[2,12],[2,13],[2,14],[2,15],[2,17],[2,18],[2,19],[2,20],[2,21],[2,23],[2,24],[2,25],[2,26],[3,4],[3,5],[3,6],[3,7],[3,8],[3,10],[3,11],[3,12],[3,13],[3,14],[3,15],[3,17],[3,18],[3,19],[3,20],[3,21],[3,23],[3,24],[3,25],[3,26],[4,10],[4,11],[4,12],[4,13],[4,14],[4,15],[4,17],[4,18],[4,19],[4,20],[4,21],[4,23],[4,24],[4,25],[4,26],[5,10],[5,11],[5,12],[5,13],[5,14],[5,15],[5,17],[5,18],[5,19],[5,20],[5,21],[5,23],[5,24],[5,25],[5,26],[6,10],[6,11],[6,12],[6,13],[6,14],[6,15],[6,17],[6,18],[6,19],[6,20],[6,21],[6,23],[6,24],[6,25],[6,26],[7,10],[7,11],[7,12],[7,13],[7,14],[7,15],[7,17],[7,18],[7,19],[7,20],[7,21],[7,23],[7,24],[7,25],[7,26],[8,10],[8,11],[8,12],[8,13],[8,14],[8,15],[8,17],[8,18],[8,19],[8,20],[8,21],[8,23],[8,24],[8,25],[8,26],[9,10],[9,11],[9,12],[9,13],[9,14],[9,15],[9,17],[9,18],[9,19],[9,20],[9,21],[9,23],[9,24],[9,25],[9,26],[10,17],[10,18],[10,19],[10,20],[10,21],[10,23],[10,24],[10,25],[10,26],[11,17],[11,18],[11,19],[11,20],[11,21],[11,23],[11,24],[11,25],[11,26],[12,17],[12,18],[12,19],[12,20],[12,21],[12,23],[12,24],[12,25],[12,26],[13,17],[13,18],[13,19],[13,20],[13,21],[13,23],[13,24],[13,25],[13,26],[14,17],[14,18],[14,19],[14,20],[14,21],[14,23],[14,24],[14,25],[14,26],[15,17],[15,18],[15,19],[15,20],[15,21],[15,23],[15,24],[15,25],[15,26],[16,17],[16,18],[16,19],[16,20],[16,21],[16,23],[16,24],[16,25],[16,26],[17,23],[17,24],[17,25],[17,26],[18,23],[18,24],[18,25],[18,26],[19,23],[19,24],[19,25],[19,26],[20,23],[20,24],[20,25],[20,26],[21,23],[21,24],[21,25],[21,26],[22,23],[22,24],[22,25],[22,26]])
    # radii for each sphere
    radii = np.array([
        0.008, 0.009, 0.0095, 0.0105,           
        0.0055, 0.007, 0.0075, 0.0082, 0.009, 0.0095,          
        0.006, 0.007, 0.0075, 0.0082, 0.009, 0.0095, 0.011,
        0.0055, 0.007, 0.0072, 0.008, 0.0085, 0.009, 
        0.005, 0.006, 0.0065, 0.007, 0.0085  
    ])


    vert_ids_per_sphere = np.array([
        [727, 763, 748, 734],
        [731, 756, 749, 733],
        [708, 754, 710, 713],
        [250, 267, 249, 28],

        [350, 314, 337, 323],
        [343, 316, 322, 336],
        [342, 295, 299, 297],
        [280, 56, 222, 155],
        [165, 133, 174, 189],
        [136, 139, 176, 170],

        [462, 426, 460, 433],
        [423, 455, 448, 432],
        [430, 454, 457, 431],
        [397, 405, 390, 398],
        [357, 364, 391, 372],
        [375, 366, 381, 367],
        [379, 399, 384, 380],

        [573, 537, 560, 544],
        [566, 534, 559, 543],
        [565, 541, 542, 523],
        [507, 476, 501, 508],
        [496, 498, 491, 495],
        [489, 509, 494, 490],

        [690, 654, 677, 664],
        [682, 658, 642, 669],
        [581, 633, 619, 629],
        [614, 616, 609, 613],
        [607, 627, 612, 608],
    ])

    return sphere_id_pairs, radii, vert_ids_per_sphere

def compute_sphere_center(v, i_v_per_sphere):
    return np.mean(v[i_v_per_sphere], axis=1)   # (28, 3)
