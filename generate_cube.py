import open3d as o3d
import os
import numpy as np

MODEL_NAME='cube_custom'

MODEL_WIDTH = 100
MODEL_HEIGHT = 100
MODEL_DEPTH = 100

os.makedirs(MODEL_NAME, exist_ok=True)

box = o3d.geometry.TriangleMesh.create_box(MODEL_WIDTH, MODEL_HEIGHT, MODEL_DEPTH).translate((-MODEL_WIDTH / 2, -MODEL_HEIGHT / 2, -MODEL_DEPTH / 2))

def write_vrml(name: str, mesh: o3d.geometry.TriangleMesh, color = None):
    with open(name, 'w') as f:
        f.write('#VRML V2.0 utf8\n\n') # initial lines

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

write_vrml(f'{MODEL_NAME}/{MODEL_NAME}.wrl', box, (1, 0, 1))
with open(f'{MODEL_NAME}/{MODEL_NAME}.xml', 'w') as f:
    f.write('\n'.join([
        '<?xml version="1.0" ?>',
        '<root>',
        '\t<material>plastic</material>',
        '\t<mass>300</mass>',
        f'\t<geometryFile type="Inventor">{MODEL_NAME}.wrl</geometryFile>',
        '</root>'
    ]))

print(f'Generated a {MODEL_WIDTH}x{MODEL_HEIGHT}x{MODEL_DEPTH} mm cube at {MODEL_NAME}')