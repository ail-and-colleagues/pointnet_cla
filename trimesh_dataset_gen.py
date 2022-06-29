import os
import numpy as np
import trimesh
from pyrr import matrix44

def random_point_in_triangle(p0, p1, p2, point_num):
    """
    create random points in a triangle composed by p0, p1, p2.
    """
    t = np.random.rand(point_num, 2)
    t = [(c[0], c[1]) if c[0] + c[1] <= 1.0 else (1.0 - c[0],  1.0 - c[1])for c in t]
    t = np.array(t)
    v = [(1 - c[0] - c[1]) * p0 + c[0] * p1 + c[1] * p2 for c in t]
    return v


def generate_pointcloud(geo, point_num=512):
    areaSum = np.sum(geo.area_faces)
    num_per_area = point_num / areaSum

    pointcloud = list()
    for f, area in zip(geo.faces, geo.area_faces):
        t_num = int(num_per_area * area)
        # +1 to avoid 0
        t_num += 1
        p0, p1, p2 = geo.vertices[f]
        v = random_point_in_triangle(p0, p1, p2, t_num)
        pointcloud += v
    pointcloud = np.array(pointcloud)
    idx = np.arange(pointcloud.shape[0])
    idx = np.random.choice(idx, point_num, replace=False)
    return pointcloud[idx]

def create_random_rot_martix():
    deg = 2.0 * np.pi * np.random.random()
    axis = np.random.uniform(-1, 1, 3)
    axis = axis / np.linalg.norm(axis)
    rm = matrix44.create_from_axis_rotation(axis=axis, theta=deg).T
    return rm

def create_box(label, prex, i, point_num, dst):
    mx = 2.0
    mn = 1.0
    w, d, h = (mx - mn) * np.random.random(3) + mn
    geo = trimesh.primitives.Box(extents=(w, d, h))
    pcd = generate_pointcloud(geo, point_num)
    pcd = pcd - np.mean(pcd, axis=0)
    pcd = trimesh.points.PointCloud(pcd)
    rm = create_random_rot_martix()
    pcd.apply_transform(rm)
    dst = os.path.join(dst, label, "{}_{:04d}.ply".format(prex, i))
    pcd.export(dst)

def create_capsule(label, prex, i, point_num, dst):
    mx = 2.0
    mn = 1.0
    r, h = (mx - mn) * np.random.random(2) + mn
    r *= 0.5
    h += 0.75
    geo = trimesh.primitives.Capsule(radius=r, height=h)
    pcd = generate_pointcloud(geo, point_num)
    pcd = pcd - np.mean(pcd, axis=0)
    pcd = trimesh.points.PointCloud(pcd)
    rm = create_random_rot_martix()
    pcd.apply_transform(rm)
    dst = os.path.join(dst, label, "{}_{:04d}.ply".format(prex, i))
    pcd.export(dst)

def create_cylinder(label, prex, i, point_num, dst):
    mx = 2.0
    mn = 1.0
    r, h = (mx - mn) * np.random.random(2) + mn
    r *= 0.5

    geo = trimesh.primitives.Cylinder(radius=r, height=h)
    pcd = generate_pointcloud(geo, point_num)
    pcd = pcd - np.mean(pcd, axis=0)
    pcd = trimesh.points.PointCloud(pcd)
    rm = create_random_rot_martix()
    pcd.apply_transform(rm)
    dst = os.path.join(dst, label, "{}_{:04d}.ply".format(prex, i))
    pcd.export(dst)

def create_scaled_sphere(label, prex, i, point_num, dst):
    mx = 2.0
    mn = 1.0
    r = (mx - mn) * np.random.random(1) + mn
    r *= 0.5

    mx = 1.5
    mn = 0.75
    sx, sy = (mx - mn) * np.random.random(2) + mn

    geo = trimesh.primitives.Sphere(radius=r)
    pcd = generate_pointcloud(geo, point_num)
    pcd = pcd - np.mean(pcd, axis=0)
    pcd[:, 0] = sx * pcd[:, 0]
    pcd[:, 1] = sy * pcd[:, 1]
    pcd = trimesh.points.PointCloud(pcd)
    rm = create_random_rot_martix()
    pcd.apply_transform(rm)
    dst = os.path.join(dst, label, "{}_{:04d}.ply".format(prex, i))
    pcd.export(dst)

if __name__ == "__main__":
    dataset_name = "trimesh_primitives"
    primitives = {
        "box": create_box,
        "capsule": create_capsule,
        "cylinder": create_cylinder, 
        "scaled_sphere": create_scaled_sphere}
    
    data_num = 10
    point_num = 512
    print(dataset_name)
    print(os.path.join("./dataset/", dataset_name))
    os.makedirs(os.path.join("./dataset", dataset_name), exist_ok=True)
    for i, p in enumerate(primitives):
        os.makedirs(os.path.join("./dataset", dataset_name, str(i)), exist_ok=True)
    
    dataset_dir = os.path.join("./dataset", dataset_name)
    for i in range(data_num):
        for l, (k, f) in enumerate(primitives.items()):            
            f(str(l), k, i, point_num, dataset_dir)

