from itertools import chain
import scipy.spatial
import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np
import yaml
import os
import time
import shutil
import xml.etree.ElementTree as ET
import open3d as o3d

DEXYCB_PATH = os.environ.get('DEX_YCB_DIR', 'D:/dexycb')
DEXYCB_SEQUENCE = '20201015-subject-09/20201015_142601'
DEXYCB_CAMERA = '932122062010'; DEXYCB_FRAME = 55

URDF_FILE = 'mano_20200514_142106_subject-09_right/mano_20200514_142106_subject-09_right.urdf'

MANO_MODELS_PATH = os.environ.get('MANO_MODELS_PATH', 'D:/mano/models')

with open(f'{DEXYCB_PATH}/{DEXYCB_SEQUENCE}/meta.yml') as f: DEXYCB_SUBJECT = yaml.safe_load(f)['mano_calib'][0]
with open(f'{DEXYCB_PATH}/calibration/mano_{DEXYCB_SUBJECT}/mano.yml') as f: mano_betas = yaml.safe_load(f)['betas']
mano_betas = torch.from_numpy(np.array([mano_betas], dtype=np.float32))

TEST = False
if TEST:
    mano_layer = ManoLayer(mano_root=MANO_MODELS_PATH, use_pca=True, ncomps=45, side=DEXYCB_SUBJECT.split('_')[-1], flat_hand_mean=True) # DexYCB uses 45 PCA components; flat_hand_mean=False is used for DexYCB

    # create hand
    hand_verts, hand_joints = mano_layer(torch.zeros(1, 45 + 3), mano_betas)
else:
    mano_layer = ManoLayer(mano_root=MANO_MODELS_PATH, use_pca=True, ncomps=45, side=DEXYCB_SUBJECT.split('_')[-1], flat_hand_mean=False) # DexYCB uses 45 PCA components; flat_hand_mean=False is used for DexYCB

    # create hand
    pose_m = torch.from_numpy(np.array([np.hstack((np.load(f'{DEXYCB_PATH}/{DEXYCB_SEQUENCE}/{DEXYCB_CAMERA}/labels_{DEXYCB_FRAME:06d}.npz')['pose_m'][0,:45], np.zeros(3)))], dtype=np.float32))
    hand_verts, hand_joints = mano_layer(pose_m, mano_betas)

# demo.display_hand({'verts': hand_verts, 'joints': hand_joints}, mano_faces=mano_layer.th_faces)
# hand_verts /= 1000; hand_joints /= 1000 # important (for DexYCB at least - TODO: test with RGB-to-MANO)

# hand_verts = hand_verts[0].detach().cpu().numpy() # first one in batch
hand_joints = hand_joints[0].detach().cpu().numpy()

# translate vertices and joints to centre at wrist
wrist_pos = hand_joints[0]
# hand_verts -= wrist_pos
hand_joints -= wrist_pos

urdf = ET.parse(URDF_FILE).getroot() # read URDF file
def read_joint(name):
    return urdf.find(f"./joint[@name='{name}']")
def read_link(name):
    return urdf.find(f"./link[@name='{name}']")

def xyz_to_array(s):
    return np.array([float(x) for x in s.split()])

# list of joints in sequential order
URDF_JOINTS = {
    'thumb': [(None, '0f', (1, 2)), ('0f', '0a', (1, 2)), ('0a', '1', (2, 3)), ('1', '2', (3, 4)), ('2', '3', (4, 5))],
    'index': [(None, '0a', (5, 6)), ('0a', '0f', (5, 6)), ('0f', '1', (6, 7)), ('1', '2', (7, 8)), ('2', '3', (8, 9))],
    'mid': [(None, '0a', (9, 10)), ('0a', '0f', (9, 10)), ('0f', '1', (10, 11)), ('1', '2', (11, 12)), ('2', '3', (12, 13))],
    'ring': [(None, '0a', (13, 14)), ('0a', '0f', (13, 14)), ('0f', '1', (14, 15)), ('1', '2', (15, 16)), ('2', '3', (16, 17))],
    'pinky': [(None, '0a', (17, 18)), ('0a', '0f', (17, 18)), ('0f', '1', (18, 19)), ('1', '2', (19, 20)), ('2', '3', (17, 18))]
}

model_finger_vectors = np.array([xyz_to_array(read_joint(f'palm_{finger}{URDF_JOINTS[finger][0][1]}').find('./origin').attrib['xyz']) for finger in URDF_JOINTS])

finger_vectors = np.array([hand_joints[URDF_JOINTS[finger][0][2][0]] - hand_joints[0] for finger in URDF_JOINTS])
rot_matrix, rssd = scipy.spatial.transform.Rotation.align_vectors(model_finger_vectors, finger_vectors)
rot_matrix = rot_matrix.as_matrix()
hand_joints = rot_matrix.dot(hand_joints.T).T # rotate to align with reference

# calculate angle from a to b, given the rotational axis
def calc_rot_angle(a, b, rot):
    # project a and b onto plane with rot as normal
    a -= a.dot(rot) * rot
    b -= b.dot(rot) * rot

    # calculate rotation angle
    ang = np.arccos(a.dot(b) / np.sqrt(a.dot(a) * b.dot(b)))
    if np.cross(a, b).dot(rot) < 0: ang *= -1 # consider rotation direction

    return ang

def joint_name(finger, tup):
    a = tup[0]; b = tup[1]
    a = 'palm' if a is None else f'{finger}{a}'
    b = f'{finger}{b}'
    return f'{a}_{b}'

def make_4x4trans(rot: np.ndarray = np.eye(3), trans: np.ndarray = np.zeros(3)):
    return np.vstack((
        np.hstack((rot, trans.reshape((3, 1)))),
        np.array([0, 0, 0, 1])
    ))

# https://stackoverflow.com/a/59204638
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if np.all(np.isclose(kmat, np.zeros((3, 3)))): return np.eye(3)

    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotation_matrix(x, y, z) -> np.array:
    sx, sy, sz = np.sin([x, y, z])
    cx, cy, cz = np.cos([x, y, z])
    return np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy, cy*sx, cy*cx]
    ])

graspit_dof_bases = {'index': 0, 'mid': 4, 'pinky': 8, 'ring': 12, 'thumb': 16} # DOF index bases

graspit_dofs = [0] * 4 * 5
# frame_meshes = []
# iterate through each finger and calculate joint angles
for n, finger in enumerate(URDF_JOINTS.keys()):
    frame = np.eye(3) # rotation only
    print(f'Processing finger {finger} (index {n}).')
    for i in range(len(URDF_JOINTS[finger]) - 1):
        joint = URDF_JOINTS[finger][i]
        # frame_meshes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=20).rotate(frame, center=(0,0,0)).translate(hand_joints[joint[2][0]]))
        joint_elem = read_joint(joint_name(finger, joint))

        # assert(np.all(np.isclose(frame[:3,3], hand_joints[joint[2][0]])))

        ref_vect = None
        for j in range(i + 1, len(URDF_JOINTS[finger])):
            joint_end = read_joint(joint_name(finger, URDF_JOINTS[finger][j])).find('./origin')
            if joint_end is None: continue
            ref_vect = xyz_to_array(joint_end.attrib['xyz'])
            break
        if ref_vect is None:
            raise RuntimeError('cannot form ref_vect')
            
        rot_axis = xyz_to_array(joint_elem.find('./axis').attrib['xyz']) # rotational axis

        # calculate rotation angle (in joint frame)
        vect = np.linalg.inv(frame).dot(hand_joints[joint[2][1]] - hand_joints[joint[2][0]])
        ref_vect /= np.sqrt(ref_vect.dot(ref_vect)); vect /= np.sqrt(vect.dot(vect))
        ang = calc_rot_angle(ref_vect, vect, rot_axis)

        print(f' - Joint {i} ({joint[0]} -> {joint[1]}): {np.degrees(ang)} deg')

        # transform joint frame to next one
        rot_align = rotation_matrix_from_vectors(np.array([0, 0, 1]), rot_axis) # bring rotation axis to Z axis
        frame_rot = rot_align.dot(rotation_matrix(0, 0, ang)).dot(np.linalg.inv(rot_align)) # rotation matrix to align current frame to next frame
        
        frame = frame.dot(frame_rot)
        # frame = frame.dot(frame).dot(make_4x4trans(frame_rot, frame_trans)).dot(np.linalg.inv(frame))

        graspit_dofs[graspit_dof_bases[finger] + i] = ang.item()
    
print('DoFs: ' + ','.join([str(x) for x in graspit_dofs]))

# o3d.visualization.draw_geometries(frame_meshes)
