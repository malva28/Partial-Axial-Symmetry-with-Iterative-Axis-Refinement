import numpy as np
from copy import deepcopy
from progress_bar import print_progress_bar


def distance_to_z_axis(p):
    return np.sqrt(p[0]**2 + p[1]**2)


def binary_search_first_geq(array, value):
    """
    Finds the index of the first element greater or equal to a desired value
    :param array: array/list
    :param value: value to find
    :return:
    """

    def recursive_binary_search_first_geq(arr, val, l_index, r_index):
        if r_index < l_index:
            if l_index >= len(arr):
                return -1
            return l_index
        m_index = l_index + (r_index - l_index) // 2

        if arr[m_index] < val:
            return recursive_binary_search_first_geq(arr, val, m_index + 1, r_index)
        else:
            return recursive_binary_search_first_geq(arr, val, l_index, m_index - 1)

    return recursive_binary_search_first_geq(array, value, 0, len(array) - 1)


def sort_points_in_z_axis(points, face_vertex_indices):
    sorted_indices = np.argsort(points[:, 2])
    inverse_sorted_indices = np.argsort(sorted_indices)
    sorted_points = deepcopy(points)[sorted_indices]

    sorted_fvi = deepcopy(face_vertex_indices)
    for i in range(len(face_vertex_indices)):
        for j in range(len(face_vertex_indices[i])):
            sorted_fvi[i][j] = inverse_sorted_indices[face_vertex_indices[i][j]]

    return sorted_points, sorted_fvi


def compute_symmetry_count_scalar_quantity(sorted_points, threshold=0.01):
    """
    Computes an array with the amount of symmetric points each point has around the z-axis.
    A point is axial-symmetric to another if they are around at the same distance to z and by the same z coord

    :param sorted_points: Points must be sorted in z-axis
    :param threshold: to consider distances close enough to be symmetric
    :return: numpy array
    """
    symmetry_count = [0]*len(sorted_points)

    for i in range(len(sorted_points)):
        point = sorted_points[i]

        bottom = binary_search_first_geq(sorted_points[:, 2], point[2] - threshold)
        top = binary_search_first_geq(sorted_points[:, 2], point[2] + threshold)

        d_to_axis_point = distance_to_z_axis(point)
        for neighbor_point in sorted_points[bottom: top]:
            d_to_axis_neighbor = distance_to_z_axis(neighbor_point)
            if abs(d_to_axis_point - d_to_axis_neighbor) < threshold:
                symmetry_count[i] += 1
        print_progress_bar(i + 1, len(sorted_points), prefix='Progress:', length=20)

    return np.array(symmetry_count)
