import numpy as np
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from transformations import rotation_axis
import torch
import openmesh


def compute_symmetry_chamfer_distance_from_mesh(mesh: openmesh.TriMesh,
                                                generator_circle: "Circle",
                                                sample_size=1000,
                                                n_angles: int = 12):
    point_cloud = mesh.points()
    verts = torch.tensor(point_cloud, dtype=torch.float32)
    faces = torch.tensor([[fi.idx() for fi in mesh.fv(face)] for face in mesh.faces()], dtype=torch.float32)
    trg_mesh = Meshes(verts=[verts], faces=[faces])
    # this method applies uniform sampling, but proportional to face area
    point_cloud_tensor = sample_points_from_meshes(trg_mesh, sample_size)
    loss = compute_symmetry_chamfer_distance(point_cloud_tensor, generator_circle, n_angles)
    return loss, point_cloud_tensor


def compute_symmetry_chamfer_distance(point_cloud: torch.Tensor, generator_circle: "Circle", n_angles: int = 12):
    angles = np.linspace(0, 2*np.pi, num=n_angles, endpoint=False)
    loss_array = np.zeros(angles.shape)
    for i in range(len(angles)):
        theta = angles[i]
        point_1 = generator_circle.c
        point_2 = point_1 + generator_circle.n
        rot_mat = rotation_axis(theta, point_1, point_2)

        rotation_tensor = torch.tensor(rot_mat, dtype=torch.float32)
        rotation_tensor = rotation_tensor.unsqueeze(0)
        transformed_points_tensor = torch.bmm(point_cloud, rotation_tensor[:, :3, :3])

        loss, _ = chamfer_distance(point_cloud, transformed_points_tensor, batch_reduction="sum")
        loss_array[i] = loss
    return loss_array.sum()
