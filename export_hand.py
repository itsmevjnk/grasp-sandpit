import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import open3d as o3d
import numpy as np
import yaml
import os
import time

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
hand_verts /= 1000; hand_joints /= 1000 # important (for DexYCB at least - TODO: test with RGB-to-MANO)
verts_np = hand_verts[0].detach().cpu().numpy() # first one in batch
hand_mesh = o3d.geometry.TriangleMesh()
# hand_mesh = hand_mesh.translate(labels['pose_m'][0, 48:51])
hand_mesh.vertices = o3d.utility.Vector3dVector(verts_np)
hand_mesh.triangles = o3d.utility.Vector3iVector(faces_np)
hand_mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([o3d.geometry.LineSet.create_from_triangle_mesh(hand_mesh)])

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

# show each segment
seg_meshes = []
for seg in colours:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = hand_mesh.vertices
    mesh.triangles = o3d.utility.Vector3iVector(faces_np[colours[seg]])
    mesh.compute_vertex_normals(); mesh.paint_uniform_color(palette[seg])
    seg_meshes.append(mesh)

o3d.visualization.draw_geometries(seg_meshes)

# colours:
# 0 = ring MCP-PIP
# 1 = index PIP-DIP
# 2 = pinky MCP-PIP
# 3 = middle MCP-PIP
# 4 = middle DIP-TIP
# 5 = ring DIP-TIP
# 6 = pinky DIP-TIP
# 7 = palm
# 8 = thumb CMC-MCP
# 9 = thumb MCP-IP
# 10 = index MCP-PIP
# 11 = index DIP-TIP
# 12 = thumb IP-TIP
# 13 = pinky PIP-DIP
# 14 = middle PIP-DIP
# 15 = ring PIP-DIP
