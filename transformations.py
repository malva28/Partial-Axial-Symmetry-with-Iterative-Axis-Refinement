
import numpy as np
from supporting_circles import Circle
import functools


# Esta función calcula el ángulo entre dos aristas u y v
def myangle(u, v):
    du = np.linalg.norm(u)
    dv = np.linalg.norm(v)

    du = max(du, 1e-8)
    dv = max(dv, 1e-8)

    return np.arccos(np.dot(u, v) / (du * dv))


def calculate_angle_diffs_with_vector(vec_arr, a_vec):
    return np.array([myangle(row, a_vec) for row in vec_arr])



def get_bounding_box_extremes(points):
    # Find box-hull diagonal extremes
    (min_x, max_x) = (np.infty, -np.infty)
    (min_y, max_y) = (np.infty, -np.infty)
    (min_z, max_z) = (np.infty, -np.infty)
    for point in points:
        min_x = point[0] if point[0] < min_x else min_x
        min_y = point[1] if point[1] < min_y else min_y
        min_z = point[2] if point[2] < min_z else min_z

        max_x = point[0] if point[0] > max_x else max_x
        max_y = point[1] if point[1] > max_y else max_y
        max_z = point[2] if point[2] > max_z else max_z

    return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])


def normalize(points):
    extreme_1, extreme_2 = get_bounding_box_extremes(points)
    # re-center to (0, 0, 0)
    recenter(points, (extreme_1 + extreme_2) / 2)

    # Scale by 1/box_diagonal
    distance = np.linalg.norm(extreme_1 - extreme_2)
    for point in points:
        point /= distance


def recenter(points, new_center):
    for point in points:
        point -= new_center


def rotate(points, rotation_matrix):
    for point in points:
        point[:] = np.matmul(rotation_matrix, point)


def reorient_point_cloud_by_angles(points, phi, theta):
    rotation_z = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    rotation_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    rotate(points, np.matmul(rotation_y, rotation_z))


def reorient_point_cloud(points, axial_circle: Circle):
    recenter(points, axial_circle.c)

    phi = -axial_circle.get_phi()
    theta = -axial_circle.get_theta()
    reorient_point_cloud_by_angles(points, phi, theta)


def reorient_circle(circle, axial_circle: Circle):
    # recenter
    circle.c -= axial_circle.c

    # reorient
    phi = -axial_circle.get_phi()
    theta = -axial_circle.get_theta()
    rotation_z = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    rotation_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    circle.n = np.matmul(np.matmul(rotation_y, rotation_z), circle.n)
    circle.c = np.matmul(np.matmul(rotation_y, rotation_z), circle.c)


# funciones extraídas de transformations.py del repositorio de Computación Gráfica
def translate(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]], dtype=np.float32)


def rotation_x(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [1, 0, 0, 0],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta, 0],
        [0, 0, 0, 1]], dtype=np.float32)


def rotation_axis(theta, point1, point2):
    axis = point2 - point1
    axis = axis / np.linalg.norm(axis)
    a, b, c = axis
    h = np.sqrt(a ** 2 + c ** 2)

    t = translate(-point1[0], -point1[1], -point1[2])
    tinv = translate(point1[0], point1[1], point1[2])

    ry = np.array([
        [a / h, 0, c / h, 0],
        [0, 1, 0, 0],
        [-c / h, 0, a / h, 0],
        [0, 0, 0, 1]], dtype=np.float32)

    ryinv = np.array([
        [a / h, 0, -c / h, 0],
        [0, 1, 0, 0],
        [c / h, 0, a / h, 0],
        [0, 0, 0, 1]], dtype=np.float32)

    rz = np.array([
        [h, b, 0, 0],
        [-b, h, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)

    rzinv = np.array([
        [h, -b, 0, 0],
        [b, h, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)

    rx = rotation_x(theta)

    return functools.reduce(np.matmul, [tinv, ryinv, rzinv, rx, rz, ry, t])
