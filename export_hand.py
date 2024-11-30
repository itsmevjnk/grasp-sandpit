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
verts_np = hand_verts[0].detach().cpu().numpy() # first one in batch
hand_mesh = o3d.geometry.TriangleMesh()
# hand_mesh = hand_mesh.translate(labels['pose_m'][0, 48:51])
hand_mesh.vertices = o3d.utility.Vector3dVector(verts_np)
hand_mesh.triangles = o3d.utility.Vector3iVector(faces_np)
hand_mesh.compute_vertex_normals()

# hand_wire = o3d.geometry.LineSet.create_from_triangle_mesh(hand_mesh)
# o3d.visualization.draw_geometries([hand_wire])

# joint_annotations = []
# for joint in range(21):
#     print(f'showing joint {joint}')
#     o3d.visualization.draw_geometries([hand_wire, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=hand_joints[0,joint])])

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

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
# show each segment
seg_meshes = []
for seg in colours:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = hand_mesh.vertices
    mesh.triangles = o3d.utility.Vector3iVector(faces_np[colours[seg]])
    mesh.remove_unreferenced_vertices() # clean up vertices
    mesh.compute_vertex_normals(); mesh.paint_uniform_color(palette[seg])
    seg_meshes.append(mesh)

# o3d.visualization.draw_geometries(seg_meshes + [frame])

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

# frame_meshes = []
# seg_wires = []
# for seg, joint in seg_map:
#     seg_wires.append(o3d.geometry.LineSet.create_from_triangle_mesh(seg_meshes[seg]))
#     frame_meshes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=hand_joints[0,joint]))

# o3d.visualization.draw_geometries(seg_wires + frame_meshes + [frame])

# demo.display_hand({'verts': hand_verts, 'joints': hand_joints}, mano_faces=mano_layer.th_faces)

def write_vrml(name: str, mesh: o3d.geometry.TriangleMesh, color = None):
    with open(name, 'w') as f:
        f.write('#VRML V2.0 utf8\n#material plastic\n#mass 150.0\n\n') # initial lines

        f.write('Shape {\n')
        
        if color:
            f.write('\tappearance Appearance {\n\t\tmaterial Material {\n\t\t\tdiffuseColor ' + ' '.join(str(x) for x in color) + '\n\t\t}\n\t}\n')

        f.write('\tgeometry IndexedFaceSet {\n')

        # coord Coordinate
        f.write('\t\tcoord Coordinate {\n\t\t\tpoint [\n')
        for point in np.asarray(mesh.vertices).tolist():
            f.write('\t\t\t\t' + ' '.join(str(x) for x in point) + '\n')
        f.write('\t\t\t]\n\t\t}\n\t\tcoordIndex [\n')
        for triangle in np.asarray(mesh.triangles).tolist():
            f.write('\t\t\t' + ' '.join(str(x) for x in triangle) + ' -1\n')
        f.write('\t\t]\n\t\tnormal Normal {\n\t\t\tvector [\n')
        for vect in np.asarray(mesh.vertex_normals).tolist():
            f.write('\t\t\t\t' + ' '.join(str(x) for x in vect) + '\n')
        f.write('\t\t\t]\n\t\t}\n')

        f.write('\t}\n}\n')

os.makedirs(f'{DEXYCB_SUBJECT}/iv', exist_ok=True)

# copy iv xml files
for seg in seg_names: shutil.copy(f'iv_xmls/{seg}.xml', f'{DEXYCB_SUBJECT}/iv/')


def rotation_matrix(x, y, z) -> np.array:
    sx, sy, sz = np.sin([x, y, z])
    cx, cy, cz = np.cos([x, y, z])
    return np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy, cy*sx, cy*cx]
    ])

finger_bases = {'thumb': 1, 'index': 5, 'mid': 9, 'ring': 13, 'pinky': 17}
joint_to_finger_idx = \
    [None] \
    + list(chain(*[[(finger, i) for i in range(4)] for finger in finger_bases]))

# finger_align_mats = {name: [] for name in finger_bases} # matrix for aligning each finger to X-

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
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

rx = rotation_matrix(np.pi, 0, 0) # rotation matrix to use below

# create rotated joint positions
hand_joints_rot = rx.dot(hand_joints[0].detach().cpu().numpy().T).T
# hand_joints_rot -= hand_joints_rot[0]

# fields for writing into descriptor
from collections import defaultdict
fields = defaultdict(lambda: '0')

# translate hand parts back and generate VRML
for seg, joint in seg_map:
    pos = hand_joints[0,joint]
    # print(f'seg {seg} ({seg_names[seg]}, joint {joint}) is at {pos}, current centre: {seg_meshes[seg].get_center()}')
    if joint != 0: # rotate hand part
        seg_meshes[seg].rotate(rx, center=(0, 0, 0))
        pos = rx.dot(pos)
    seg_meshes[seg].translate(-pos)
    if joint != 0:
        finger, idx = joint_to_finger_idx[joint]
        if finger != 'thumb': # TODO: thumb
            # straighten fingers (by aligning their vectors to X-)
            vect = hand_joints_rot[joint + 1] - hand_joints_rot[joint]
            straight_rot = rotation_matrix_from_vectors(vect, np.array([-1, 0, 0]))
            seg_meshes[seg].rotate(straight_rot, center=(0, 0, 0))

            length = np.sqrt(vect.dot(vect)) # finger length
            seg_meshes[seg].translate((length, 0, 0)) # move fingers back towards palm

            fields[f'{finger}{idx}_r'] = f'{-length}'

            # if finger == 'pinky':
            #     o3d.visualization.draw_geometries([seg_meshes[seg], frame])
    # if joint != 0: seg_meshes[seg].rotate(seg_meshes[seg].get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))) # flip
    # print(f' - new centre: {seg_meshes[seg].get_center()}')
    # o3d.io.write_triangle_mesh(f'{seg_names[seg]}.obj', seg_meshes[seg], print_progress=True)
    write_vrml(f'{DEXYCB_SUBJECT}/iv/{seg_names[seg]}.wrl', seg_meshes[seg], palette[seg])

# calculate translation and rotation matrix for fingers (and also DH params)
wrist_pos = hand_joints[0, 0].clone(); hand_joints[0] -= wrist_pos # centre about wrist (as with 3D model)
base_rot = ' '.join(f'{x:.18f}' for x in rotation_matrix(-np.pi / 2, 0, 0).reshape(-1).tolist()) # rotation matrix for chain base
for finger in finger_bases: # NOTE: last matrix is the first transformation!
    mcp = finger_bases[finger]
    fields[f'{finger}T'] = ' '.join(f'{x:.18f}' for x in (hand_joints[0, mcp]).tolist())
    fields[f'{finger}R'] = base_rot

    if finger != 'thumb': # TODO: thumb
        origin = hand_joints[0, mcp] # origin of link
        for i in range(3): # 3 links: MCP-PIP, PIP-DIP and DIP-TIP
            vect = hand_joints[0, i + 1] - hand_joints[0, i] # link pointer vector
            vect = rotation_matrix_from_vectors(vect, np.array([-1, 0, 0])).dot(vect) # straighten (as we just did above)
            end = origin + vect # end point (e.g. PIP for MCP-PIP, whre origin is MCP)
            

    # mats = finger_align_mats[finger]
    # for i in range(3):
    #     vect = hand_joints[0, mcp + i + 1] - hand_joints[0, mcp + i] # MCP-PIP, PIP-DIP, DIP-TIP (or CMC-MCP, MCP-IP, IP-TIP for thumb)
    #     mats.append(rotation_matrix_from_vectors(vect, np.array([-1, 0, 0])))

# write descriptor
with open('descriptor_template.xml', 'r') as f: template = f.read()
with open(f'{DEXYCB_SUBJECT}/{DEXYCB_SUBJECT}.xml', 'w') as f: f.write(template.format_map(fields))

# o3d.visualization.draw_geometries(seg_meshes + [frame])
