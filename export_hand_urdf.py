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

faces_np = mano_layer.th_faces.detach().cpu().numpy()
sealed_faces = np.load('sealed_faces.npy', allow_pickle=True).item() # NOTE: this only supports right hand!
# faces_np = sealed_faces['sealed_faces_right'] # they're the same, but the one from sealed_faces is sealed (first 1538 faces are the same)

# generate list of face indices matching colours
face_colours = sealed_faces['sealed_faces_color_right'][:len(faces_np)]
colours = { id: np.where(face_colours == id) for id in np.unique(face_colours) }

# create hand
pose_m = torch.zeros(1, 45 + 3)
hand_verts, hand_joints = mano_layer(pose_m, torch.from_numpy(np.array([mano_betas], dtype=np.float32)))
# demo.display_hand({'verts': hand_verts, 'joints': hand_joints}, mano_faces=mano_layer.th_faces)
# hand_verts /= 1000; hand_joints /= 1000 # important (for DexYCB at least - TODO: test with RGB-to-MANO)

hand_verts = hand_verts[0].detach().cpu().numpy() # first one in batch
hand_joints = hand_joints[0].detach().cpu().numpy()

hand_joints /= 100

finger_bases = {'thumb': 1, 'index': 5, 'mid': 9, 'ring': 13, 'pinky': 17}

with open(f'{DEXYCB_SUBJECT}.urdf', 'w') as f:
    f.write(f'<?xml version="1.0"?>\n<robot name="{DEXYCB_SUBJECT}">\n')

    links = [
        'palm',
        'index0b', 'index0a', 'index0f', 'index1', 'index2', 'index3',
        'mid0b', 'mid0a', 'mid0f', 'mid1', 'mid2', 'mid3',
        'ring0b', 'ring0a', 'ring0f', 'ring1', 'ring2', 'mid3',
        'pinky0b', 'pinky0a', 'pinky0f', 'pinky1', 'pinky2', 'mid3',
        'thumb0b', 'thumb0f', 'thumb0a', 'thumb1', 'thumb2', 'thumb3'
    ]
    for link in links: f.write(f'<link name="{link}"><geometry><sphere radius="0.5"/></geometry></link>\n') # TODO: geometry

    for finger in finger_bases:
        prev_origin = hand_joints[0] # previous origin - we start from wrist
        prev_joint = 'palm'

        for i in range(4):
            joint = f'{finger}{i}' # joint name
            origin = hand_joints[finger_bases[finger] + i]; origin_offset = origin - prev_origin

            is_thumb = finger == 'thumb'
            if i == 0: # first joint - we actually have 2 joints, a(dduction/bduction) and f(lex), and also the fixed base
                joint += 'b' # base joint
                f.write(f'<joint name="{prev_joint}_{joint}" type="fixed">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                f.write('</joint>\n')

                prev_joint = joint

                if is_thumb: # thumb would have flex before adduction
                    joint = f'{finger}{i}f' # flex
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/><origin xyz="0 0 0"/>\n<axis xyz="-1 0 0"/></joint>\n')
                else:
                    joint = f'{finger}{i}a' # abduction
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/><origin xyz="0 0 0"/>\n<axis xyz="0 -1 0"/></joint>\n')

                prev_joint = joint

                if is_thumb:
                    joint = f'{finger}{i}a'
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/><origin xyz="0 0 0"/>\n<axis xyz="0 -1 0"/></joint>\n')
                else:
                    joint = f'{finger}{i}f'
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>\n')
            elif i == 3: # last joint (tip)
                f.write(f'<joint name="{prev_joint}_{joint}" type="fixed">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                f.write('</joint>\n')
            else: # next joints
                if finger != 'thumb': # next joints are simpler
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                    f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                    f.write(f'<axis xyz="0 0 1"/></joint>\n')
                else: # CMC-MCP / MCP-IP
                    cmc_mcp = hand_joints[finger_bases[finger] + i + 1] - origin # CMC-MCP vector
                    rot_axis = np.cross(cmc_mcp, np.array([0, -1, 0])); rot_axis /= np.sqrt(rot_axis.dot(rot_axis)) # rotation axis
                    f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                    f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                    f.write(f'<axis xyz="' + ' '.join(f'{x:.18f}' for x in rot_axis.tolist()) + '"/></joint>\n')

            prev_joint = joint
            prev_origin = origin

    f.write('</robot>\n')

