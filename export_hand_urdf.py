from itertools import chain
import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import open3d as o3d
import numpy as np
import yaml
import os
import time
import shutil

DEXYCB_PATH = os.environ.get('DEX_YCB_DIR', 'D:/dexycb')
DEXYCB_SUBJECT = 'mano_20200514_142106_subject-09_right'

MANO_MODELS_PATH = os.environ.get('MANO_MODELS_PATH', 'D:/mano/models')

with open(f'{DEXYCB_PATH}/calibration/{DEXYCB_SUBJECT}/mano.yml') as f: mano_betas = yaml.safe_load(f)['betas']

mano_layer = ManoLayer(mano_root=MANO_MODELS_PATH, use_pca=True, ncomps=45, side=DEXYCB_SUBJECT.split('_')[-1]) # DexYCB uses 45 PCA components; flat_hand_mean=True by default which creates flat hand for zero thetas

faces = mano_layer.th_faces.detach().cpu().numpy()
sealed_faces = np.load('sealed_faces.npy', allow_pickle=True).item() # NOTE: this only supports right hand!
# faces_np = sealed_faces['sealed_faces_right'] # they're the same, but the one from sealed_faces is sealed (first 1538 faces are the same)

# generate list of face indices matching colours
face_colours = sealed_faces['sealed_faces_color_right'][:len(faces)]
colours = { id: np.where(face_colours == id) for id in np.unique(face_colours) }

# create hand
pose_m = torch.zeros(1, 45 + 3)
hand_verts, hand_joints = mano_layer(pose_m, torch.from_numpy(np.array([mano_betas], dtype=np.float32)))
# demo.display_hand({'verts': hand_verts, 'joints': hand_joints}, mano_faces=mano_layer.th_faces)
# hand_verts /= 1000; hand_joints /= 1000 # important (for DexYCB at least - TODO: test with RGB-to-MANO)

hand_verts = hand_verts[0].detach().cpu().numpy() # first one in batch
hand_joints = hand_joints[0].detach().cpu().numpy()

# translate vertices and joints to centre at wrist
wrist_pos = hand_joints[0]
hand_verts -= wrist_pos; hand_joints -= wrist_pos

np.save(f'{DEXYCB_SUBJECT}/zeros.npy', hand_joints) # zero positions

hand_verts /= 1000; hand_joints /= 1000 # convert to metres

finger_bases = {'thumb': 1, 'index': 5, 'mid': 9, 'ring': 13, 'pinky': 17}

os.makedirs(f'{DEXYCB_SUBJECT}/models', exist_ok=True)

# colour mapping
palette = [
    [x / 255 for x in bytes.fromhex(h)]
    for h in [
        '000000', '0000AA', '00AA00', '00AAAA',
        'AA0000', 'AA00AA', 'AA5500', 'AAAAAA',
        '555555', '5555FF', '55FF55', '55FFFF',
        'FF5555', 'FF55FF', 'FFFF55', 'FFFFFF'
    ]
]

# break down MANO model into segments
seg_meshes = []
for seg in colours:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces[colours[seg]])
    mesh.remove_unreferenced_vertices() # clean up vertices
    mesh.compute_vertex_normals(); mesh.paint_uniform_color(palette[seg])
    seg_meshes.append(mesh)

# colours:
# 0 = ring MCP-PIP
# 1 = index PIP-DIP
# 2 = pinky MCP-PIP
# 3 = middle MCP-PIP
# 4 = middle DIP-TIP
# 5 = ring DIP-TIP
# 6 = pinky DIP-TIP
# 7 = thumb CMC-MCP
# 8 = palm
# 9 = thumb MCP-IP
# 10 = index MCP-PIP
# 11 = index DIP-TIP
# 12 = thumb IP-TIP
# 13 = pinky PIP-DIP
# 14 = middle PIP-DIP
# 15 = ring PIP-DIP
seg_names = ['ring1', 'index2', 'pinky1', 'mid1', 'mid3', 'ring3', 'pinky3', 'thumb1', 'palm', 'thumb2', 'index1', 'index3', 'thumb3', 'pinky2', 'mid2', 'ring2']

seg_map = list(enumerate([13, 6, 17, 9, 11, 15, 19, 1, 0, 2, 5, 7, 3, 18, 10, 14]))

for seg, joint in seg_map:
    pos = hand_joints[joint]
    seg_meshes[seg].translate(-pos)
    o3d.io.write_triangle_mesh(f'{DEXYCB_SUBJECT}/models/{seg_names[seg]}.stl', seg_meshes[seg])

# write URDF file
MODEL_PKG = os.environ.get('MODEL_PKG', 'mano_urdf')
# MODELS_PATH = f'file://$(find {MODEL_PKG})/models'
MODELS_PATH = f'package://{MODEL_PKG}/models'
with open(f'{DEXYCB_SUBJECT}/{DEXYCB_SUBJECT}.urdf', 'w') as f:
    f.write(f'<?xml version="1.0"?>\n<robot name="{DEXYCB_SUBJECT}">\n')
    f.write(f'<link name="palm"><visual><geometry><mesh filename="{MODELS_PATH}/palm.stl"/></geometry></visual></link>\n')

    for finger in finger_bases:
        prev_origin = hand_joints[0] # previous origin - we start from wrist
        prev_joint = 'palm'

        for i in range(4):
            joint = f'{finger}{i}' # joint name
            origin = hand_joints[finger_bases[finger] + i]; origin_offset = origin - prev_origin

            is_thumb = finger == 'thumb'
            if i == 0: # first joint - we actually have 2 joints, a(dduction/bduction) and f(lex), and also the fixed base
                # joint += 'b' # base joint
                # f.write(f'<link name="{joint}"/>\n')
                # f.write(f'<joint name="{prev_joint}_{joint}" type="fixed">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                # f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                # f.write('</joint>\n')

                # prev_joint = joint

                if is_thumb: # thumb would have flex before adduction
                    joint = f'{finger}{i}f' # flex
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n<axis xyz="-1 0 0"/>\n')
                else:
                    joint = f'{finger}{i}a' # abduction
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n<axis xyz="0 -1 0"/>\n')

                f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/></joint>\n')
                f.write(f'<link name="{joint}"/>\n')
                prev_joint = joint

                if is_thumb:
                    joint = f'{finger}{i}a'
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n<axis xyz="0 -1 0"/></joint>\n')
                else:
                    joint = f'{finger}{i}f'
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/><axis xyz="0 0 1"/></joint>\n')
            
                f.write(f'<link name="{joint}"><visual><geometry><mesh filename="{MODELS_PATH}/{finger}1.stl"/></geometry></visual></link>\n')
            elif i == 3: # last joint (tip)
                f.write(f'<link name="{joint}"/>\n')
                f.write(f'<joint name="{prev_joint}_{joint}" type="fixed">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                f.write('</joint>\n')
            else: # next joints
                f.write(f'<link name="{joint}"><visual><geometry><mesh filename="{MODELS_PATH}/{finger}{i+1}.stl"/></geometry></visual></link>\n')
                if finger != 'thumb': # next joints are simpler
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                    f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                    f.write(f'<axis xyz="0 0 1"/></joint>\n')
                else: # CMC-MCP / MCP-IP
                    cmc_mcp = hand_joints[finger_bases[finger] + i + 1] - origin # CMC-MCP vector
                    rot_axis = np.cross(cmc_mcp, np.array([0, -1, 0])); rot_axis /= np.sqrt(rot_axis.dot(rot_axis)) # rotation axis
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                    f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                    f.write(f'<axis xyz="' + ' '.join(f'{x:.18f}' for x in rot_axis.tolist()) + '"/></joint>\n')

            prev_joint = joint
            prev_origin = origin

    f.write('</robot>\n')

