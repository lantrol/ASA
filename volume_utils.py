import math
import os
import time

import glm
import matplotlib.pyplot as plt
import napari
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm


def vowl(dim=64, size=0.16, r=0.04, R=0.045):
    volume = torch.zeros((dim, dim, dim), dtype=torch.float32)

    rotation = glm.mat4()
    rotation = glm.rotate(rotation, math.pi / 4, glm.vec3(0, 1, 0))

    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                pos = glm.vec4(
                    (x - dim // 2) * (size / dim),
                    (y - dim // 2) * (size / dim),
                    (z - dim // 2) * (size / dim),
                    1,
                )

                pos = rotation * pos

                dist = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)

                volume[x, y, z] = float(int(dist >= r and dist < R and pos.z < 0))

    return volume


def hollow_sphere(dim=64, size=0.16, r=0.04, R=0.045):
    volume = torch.zeros((dim, dim, dim), dtype=torch.float32)

    rotation = glm.mat4()
    rotation = glm.rotate(rotation, math.pi / 4, glm.vec3(0, 1, 0))

    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                pos = glm.vec4(
                    (x - dim // 2) * (size / dim),
                    (y - dim // 2) * (size / dim),
                    (z - dim // 2) * (size / dim),
                    1,
                )

                pos = rotation * pos

                dist = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)

                volume[x, y, z] = float(int(dist >= r and dist < R))

    return volume


def create_donut(dim, r, R):
    volume = torch.zeros((dim, dim, dim), dtype=torch.float32)

    lin_x = torch.arange(-dim // 2, dim // 2)
    lin_y = torch.arange(-dim // 2, dim // 2)
    lin_z = torch.arange(-dim // 2, dim // 2)

    mg_x, mg_y, mg_z = torch.meshgrid(lin_x, lin_y, lin_z)

    dists = (R - torch.sqrt(mg_x**2 + mg_y**2)) ** 2 + mg_z**2

    volume[dists < r**2] = 1
    return volume


def cube_frame(dim=64, size=0.16, r=0.04, R=0.05, rot=[0.0, 0.0, 0.0]):
    volume = torch.zeros((dim, dim, dim), dtype=torch.float32)

    rotation = glm.mat4()
    rotation = glm.rotate(rotation, rot[0], glm.vec3(0, 0, 1))
    rotation = glm.rotate(rotation, rot[1], glm.vec3(0, 1, 0))
    rotation = glm.rotate(rotation, rot[2], glm.vec3(1, 0, 0))
    rotation = glm.inverse(rotation)

    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                pos = glm.vec4(
                    (x - dim // 2) * (size / dim),
                    (y - dim // 2) * (size / dim),
                    (z - dim // 2) * (size / dim),
                    1,
                )

                pos = rotation * pos

                x_in_range = abs(pos.x) >= r and abs(pos.x) < R
                y_in_range = abs(pos.y) >= r and abs(pos.y) < R
                z_in_range = abs(pos.z) >= r and abs(pos.z) < R
                out_of_cube = abs(pos.x) > R or abs(pos.y) > R or abs(pos.z) > R

                volume[x, y, z] = int(
                    (
                        x_in_range
                        and y_in_range
                        or x_in_range
                        and z_in_range
                        or y_in_range
                        and z_in_range
                    )
                    and not out_of_cube
                )

                # volume[x, y, z] = int(not out_of_cube)

    return volume


def helix(dim=64, size=0.16, r=0.01, R=0.05, num_loops=2, rot=[0.0, 0.0, 0.0]):
    volume = torch.zeros((dim, dim, dim), dtype=torch.float32)

    rotation = glm.mat4()
    rotation = glm.rotate(rotation, rot[0], glm.vec3(1, 0, 0))
    rotation = glm.rotate(rotation, rot[1], glm.vec3(0, 1, 0))
    rotation = glm.rotate(rotation, rot[2], glm.vec3(0, 0, 1))

    rotation = glm.inverse(rotation)

    angle_per_cell = 2 * math.pi * num_loops / dim

    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                pos = glm.vec4(
                    (x - dim // 2) * (size / dim),
                    (y - dim // 2) * (size / dim),
                    (z - dim // 2) * (size / dim),
                    1,
                )

                pos = rotation * pos

                # Helix info
                helix_center = glm.rotate(
                    glm.mat4(), angle_per_cell * z, (0, 0, 1)
                ) * glm.vec4(R, 0, 0, 1)

                helix_center += glm.vec4(0, 0, pos.z, 0)

                dist_vec = helix_center - pos
                dist = np.sqrt(dist_vec.x**2 + dist_vec.y**2 + dist_vec.z**2)

                volume[x, y, z] = int(dist < r)
                # volume[x, y, z] = int(not out_of_cube)

    return volume
