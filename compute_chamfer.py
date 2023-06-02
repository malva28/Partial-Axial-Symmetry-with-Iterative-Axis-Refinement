import numpy as np
import openmesh
import polyscope as ps
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds, Meshes
from transformations import rotation_axis
import torch
from pytorch3d.ops import sample_points_from_meshes


from mesh_axis_render import show_mesh_with_partial_axis


def compute_symmetry_chamfer_distance_2(sorted_points, rotated_sorted_point, threshold=0.01):
    loss, _ = chamfer_distance(sorted_points, rotated_sorted_point, batch_reduction="sum")
    print(loss)


def compute_symmetry_chamfer_distance(point_cloud: torch.Tensor, generator_circle: "Circle", n_angles: int = 12):
#def compute_symmetry_chamfer_distance(mesh, generator_circle: "Circle"):
    #sorted_points = mesh.points()
    #verts = torch.tensor(mesh.points(), dtype=torch.float32)
    #faces = torch.tensor([[fi.idx() for fi in mesh.fv(face)] for face in mesh.faces()], dtype=torch.float32)
    #trg_mesh = Meshes(verts=[verts], faces=[faces])
    #sorted_point_cloud = sample_points_from_meshes(trg_mesh, sorted_points.shape[0])

    #sorted_points_4 = np.ones((sorted_points.shape[0], 4))
    #sorted_points_4[:,0:3] = sorted_points
    # sorted_point_cloud = Pointclouds(points=torch.from_numpy(sorted_points))

    #sorted_point_cloud = Pointclouds(points=torch.tensor(sorted_points, dtype=torch.float32))
    angles = np.linspace(0, 2*np.pi, num=n_angles, endpoint=False)
    loss_array = np.zeros(angles.shape)
    for i in range(len(angles)):
        theta = angles[i]
        point_1 = generator_circle.c
        point_2 = point_1 + generator_circle.n
        rot_mat = rotation_axis(theta, point_1, point_2)
        #rotated_points = np.matmul(sorted_points_4, rot_mat)
        # rotated_point_cloud = Pointclouds(points=torch.from_numpy(rotated_points[:,0:3]))
        #rotated_point_cloud = Pointclouds(points=torch.tensor(rotated_points[:, 0:3], dtype=torch.float32))
        #rotated_verts = torch.tensor(rotated_points[:, 0:3], dtype=torch.float32)
        #rot_trg_mesh = Meshes(verts=[rotated_verts], faces=[faces])
        #rotated_point_cloud = sample_points_from_meshes(rot_trg_mesh, sorted_points.shape[0])

        rotation_tensor = torch.tensor(rot_mat, dtype=torch.float32)
        rotation_tensor = rotation_tensor.unsqueeze(0)
        transformed_points_tensor = torch.bmm(point_cloud, rotation_tensor[:, :3, :3])

        #loss, _ = chamfer_distance(sorted_point_cloud, rotated_point_cloud, batch_reduction="sum")
        loss, _ = chamfer_distance(point_cloud, transformed_points_tensor, batch_reduction="sum")
        loss_array[i] = loss
        #ps.init()
        #ps.register_point_cloud("my point cloud", tensor_to_np_rot_points)
        #ps.show()

        #show_mesh_with_partial_axis(
        #    tensor_to_np_mesh,
        #    generator_circle,
        #    0.01,
        #    phi=0.0,
        #    theta=0.0,
        #    symmetric_support=False)
    return loss_array.sum()