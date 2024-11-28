import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import open3d as o3d
import numpy as np
import yaml
import os

DEXYCB_PATH = os.environ.get('DEX_YCB_DIR', 'D:/dexycb')
DEXYCB_DATASET = '20200709-subject-01/20200709_141754'
DEXYCB_CAMERA = '932122060861'
DEXYCB_FRAME = '000071'

MANO_MODELS_PATH = os.environ.get('MANO_MODELS_PATH', 'D:/mano/models')

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

with open(f'{DEXYCB_PATH}/{DEXYCB_DATASET}/meta.yml', 'r') as f: meta = yaml.safe_load(f)
labels = np.load(f'{DEXYCB_PATH}/{DEXYCB_DATASET}/{DEXYCB_CAMERA}/labels_{DEXYCB_FRAME}.npz')
with open(f'{DEXYCB_PATH}/calibration/mano_{meta["mano_calib"][0]}/mano.yml') as f: mano_betas = yaml.safe_load(f)['betas']

ycb_grasp_idx = meta['ycb_grasp_ind']
grasped_object = _YCB_CLASSES[meta['ycb_ids'][ycb_grasp_idx]]
hand = meta['mano_sides'][0]
print(f'Grasped object: {grasped_object}, hand: {hand}')

mano_layer = ManoLayer(mano_root=MANO_MODELS_PATH, use_pca=True, ncomps=45, flat_hand_mean=False, side=hand) # DexYCB uses 45 PCA components; flat_hand_mean=False is important here
faces_np = mano_layer.th_faces.detach().cpu().numpy()

# create object
obj_mesh = o3d.t.io.read_triangle_mesh(f'{DEXYCB_PATH}/models/{grasped_object}/textured.obj')
obj_tf = np.vstack((labels['pose_y'][ycb_grasp_idx], np.array([0, 0, 0, 1], dtype=np.float32)))
# obj_tf[1] *= -1; obj_tf[2] *= -1
obj_mesh = obj_mesh.transform(obj_tf)
obj_mesh.compute_vertex_normals()#; obj_mesh.paint_uniform_color([0, 1, 0])
# paint_uniform_color(obj_mesh, [0, 1, 0])

# o3d.visualization.draw(obj_mesh)

# create hand
pose_m = torch.from_numpy(labels['pose_m'])
hand_verts, hand_joints = mano_layer(pose_m[:, 0:48], torch.from_numpy(np.array([mano_betas], dtype=np.float32)), pose_m[:, 48:51])
# demo.display_hand({'verts': hand_verts, 'joints': hand_joints}, mano_faces=mano_layer.th_faces)
hand_verts /= 1000 # important
verts_np = hand_verts[0].detach().cpu().numpy() # first one in batch
hand_mesh = o3d.t.geometry.TriangleMesh()
# hand_mesh = hand_mesh.translate(labels['pose_m'][0, 48:51])
hand_mesh.vertex.positions = o3d.core.Tensor(verts_np)
hand_mesh.triangle.indices = o3d.core.Tensor(faces_np)
hand_mesh.compute_vertex_normals()#; hand_mesh.paint_uniform_color([0.25, 0.25, 0.25])

o3d.visualization.draw([obj_mesh, hand_mesh])

intersection = obj_mesh.boolean_intersection(hand_mesh)
# intersection.paint_uniform_color([1.0, 0.0, 0.0])
o3d.visualization.draw([intersection])
