import numpy as np

# https://stackoverflow.com/a/59204638
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    I = np.eye(3)
    v = np.cross(vec1, vec2)
    s = np.sqrt(v.dot(v))
    c = vec1.dot(vec2)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    if np.any(vx): return I + vx + vx.dot(vx) * ((1 - c) / s**2)
    else: return I # zero matrix

# find common normal of two axes given their origins and vectors
# returns (vector, intersect_1, intersect_2, normal_length)
def calc_common_normal(org1, vec1, org2, vec2):
     # calculate common normal coordinates (on z1 and z2)
        A = np.array([vec1 * vec1, -vec1 * vec2]).T
        B = org2 - org1
        X, residuals, rank, s = np.linalg.lstsq(A, B, -1)
        intersect1 = org1 + X[0] * vec1
        intersect2 = org2 + X[1] * vec2
        normal = intersect2 - intersect1; normal_len = np.sqrt(normal.dot(normal)); normal /= normal_len # normalise our normal
        return (normal, intersect1, intersect2, normal_len)

# calculate DH parameters for entire chain.
# the chain is given as a list of frames.
# this function returns a list of parameters (d, theta, a, alpha, actual_frame)
def build_chain(chain):
    parameters = [] # list of DH parameters
    
    f1 = chain[0] # preceding joint frame
    for i in range(1, len(chain)):
        f2 = chain[i] # next joint frame (world frame)
        f2_rel = np.linalg.inv(f1).dot(f2) # next joint frame relative to preceding one

        z2 = f2_rel[:3,2] # Z axis of f2_rel
        if np.all(np.isclose(z2[:2], [0, 0])): # parallel Z axes
            d = f2_rel[2, 3] # d can be taken directly
            alpha = 0 # otherwise this wouldn't have happened

            origin = f2_rel[:3,3] # next joint's origin
            x2 = np.hstack((origin[:2], [0]))
            if np.all(np.isclose(x2, [0, 0, 0])): x2 = np.array([1, 0, 0]) # coincidental origin - use last frame instead
            else: x2 /= np.sqrt(x2.dot(x2)) # common normal
            a = np.sqrt(origin[:2].dot(origin[:2]))
        else: # non-parallel Z axes
            x2, x2_i1, origin, a = calc_common_normal(np.zeros(3), np.array([0,0,1]), f2_rel[:3,3], z2) # calculate common normal

            d = x2_i1[2]

            alpha = np.arccos(np.array([0, 0, 1]).dot(z2))
            if np.cross(np.array([0, 0, 1]), z2).dot(x2) < 0: alpha *= -1 # check alignment of Z axes' cross product versus X axis

        theta = np.arccos(np.array([1, 0, 0]).dot(x2))
        if np.cross(np.array([1, 0, 0]), x2)[2] < 0: theta *= -1 # check alignment of X axes' cross product versus Z axis

        f2_rel = np.vstack((np.array([x2, np.cross(z2, x2), z2, origin]).T, np.array([0, 0, 0, 1]))) # reconstruct frame according to DH parameters
        f1 = f1.dot(f2_rel) # and add it on the existing frame to get the next frame in world frame
        parameters.append((d, theta, a, alpha, f1))

    return parameters
            

F01 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 20],
    [0, 0, 0, 1]
])

F12 = np.array([
    [-1 / np.sqrt(2), 0, 1 / np.sqrt(2), -25],
    [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 25],
    [0, 1, 0, 10],
    [0, 0, 0, 1]
])

F23 = np.array([
    [0, 0, 1, 0],
    [1, 0, 0, 20],
    [0, 1, 0, 25 * np.sqrt(2)],
    [0, 0, 0, 1]
])

F3 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 20],
    [0, 0, 0, 1]
])

matrices = [
    F01,
    F01.dot(F12),
    F01.dot(F12).dot(F23),
    F01.dot(F12).dot(F23).dot(F3)
]

np.set_printoptions(suppress=True)

result = build_chain(matrices)
for i, params in enumerate(result):
    print(f'Joint {i}: d={params[0]}, theta={params[1]} ({params[1] / np.pi} pi), a={params[2]}, alpha={params[3]} ({params[3] / np.pi} pi)')
    print(f'\tOriginal matrix:\n{matrices[i + 1]}')
    print(f'\tReconstructed matrix:\n{params[4]}')