import os
import argparse
import trimesh
from plyfile import PlyData, PlyElement
import numpy as np

# Spherical-harmonics DC component (SH degree 0). Inlined to avoid depending on
# any reconstruction repo (formerly imported from utils.sh_utils.SH2RGB).
SH_C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blender_location', type=str,
                        default='/path/to/blender-3.6.18-linux-x64/blender',
                        help='Absolute path to the Blender executable (Blender 3.6.x).')
    parser.add_argument('--obj_save_dir', type=str,
                        default='/path/to/datasets/MUGSQA_SOURCE',
                        help='Root directory containing one subfolder per source object.')
    parser.add_argument(
        "--res", nargs="+", type=int, default=[1080, 720, 480], help='number of views to be rendered'
    )
    parser.add_argument(
        "--views", nargs="+", type=int, default=[72, 36, 9], help='number of views to be rendered'
    )
    parser.add_argument(
        "--distance", nargs="+", type=int, default=[5, 2, 1], help='Distance between camera and model.'
    )
    parser.add_argument('--mode', type=int, default=1,
                        help='0: Resize objs, 1: Render imgs, 2: Render videos')
    return parser.parse_args()


def scale_obj(blender_location, obj_save_dir):
    os.system(blender_location + ' --background --python scale_blender.py -- --obj_save_dir %s' % (obj_save_dir))


def render_obj(blender_location, obj_save_dir, res, views, distance):
    # Create output path
    for idx, subfolder_name in enumerate(os.listdir(obj_save_dir)):
        print("Current obj idx: ", idx)
        subfolder_path = os.path.join(obj_save_dir, subfolder_name)
        sourcefolder_path = os.path.join(subfolder_path, 'calibrated')
        # Generate images for distortion
        if os.path.isdir(subfolder_path):
            output_subfolder = os.path.join(subfolder_path, 'distortion')
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            output_subfolder = os.path.join(output_subfolder, str(res)+'res'+'_'+str(views)+'views'+'_'+str(distance)+'distance')
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            output_subfolder = os.path.join(output_subfolder, 'render')
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            # Check if rendered
            if os.path.exists(os.path.join(output_subfolder, 'image.zip')):
                print("Exist!!! skip %s" % (subfolder_name))
                continue
            # Render current obj files
            for file_name in os.listdir(sourcefolder_path):
                if file_name.endswith(".obj"):
                    obj_path = os.path.join(sourcefolder_path, file_name)
                    # render to 2D
                    os.system(blender_location + ' --background --python render_blender.py -- --res %d --views %d --distance %f --obj_save_dir %s %s' % (res, views, distance, output_subfolder, obj_path))
    print("Finished!")


def render_ref(blender_location, obj_save_dir):
    print("Start!")
    # Create output path
    for idx, subfolder_name in enumerate(os.listdir(obj_save_dir)):
        print("Current obj idx: ", idx)
        subfolder_path = os.path.join(obj_save_dir, subfolder_name)
        sourcefolder_path = os.path.join(subfolder_path, 'calibrated')
        
        # TESTING
        output_subfolder = os.path.join(sourcefolder_path, 'render')

        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        # Render current obj files
        for file_name in os.listdir(sourcefolder_path):
            if file_name.endswith(".obj"):
                obj_path = os.path.join(sourcefolder_path, file_name)
                mesh = trimesh.load(obj_path, force='mesh')
                mesh_path = os.path.join(output_subfolder, "point_cloud.obj")
                if os.path.exists(os.path.join(output_subfolder, 'points3d_rnd.ply')):
                    print('Exist points3d_rnd.ply! Skip!')
                else:
                    mesh.export(mesh_path)
                    # Step1: Without GTPC. Sample randomly.
                    # if mesh do not exist, then randomize
                    num_pts = 100_000
                    print(f"Generating random point cloud ({num_pts})...")
                    ply_path = os.path.join(output_subfolder, "points3d_rnd.ply")
                    # We create random points inside the bounds of the synthetic Blender scenes
                    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
                    shs = np.random.random((num_pts, 3)) / 255.0
                    rgb = (shs * SH_C0 + 0.5) * 255  # SH degree-0 -> RGB
                    storePly(ply_path, xyz, rgb)
                if os.path.exists(os.path.join(output_subfolder, 'points3d_gt.ply')):
                    print('Exist points3d_gt.ply! Skip!')
                else:
                    # Step2: With GTPC. Sample from GTPC.
                    fetchObj(mesh_path, 100000)
                # Check if rendered
                if os.path.exists(os.path.join(output_subfolder, 'image.zip')):
                    print("Exist!!! skip %s" % (subfolder_name))
                else:
                    os.system(blender_location + ' --background --python render_blender.py -- --obj_save_dir %s --ref_mode True %s' % (output_subfolder, obj_path))
    print("Finished!")


def fetchObj(mesh_path, num_points):
    print("##### loading pointcloud from mesh object ######")
    mesh = trimesh.load(mesh_path)
    points, face_index = trimesh.sample.sample_surface(mesh, num_points)
    face_normals = mesh.face_normals
    # Retrieve the normals for the sampled points based on face index
    sampled_normals = face_normals[face_index]
    try:
        mesh.visual = mesh.visual.to_color()
        vertex_colors = mesh.visual.vertex_colors[:, :3]  # Ignore the alpha channel
        sampled_colors = np.zeros((num_points, 3))
        for i in range(num_points):
            # Get the indices of the vertices of the face
            face_vertices = mesh.faces[face_index[i]]
            # Get the vertex coordinates and their corresponding colors
            vertices = mesh.vertices[face_vertices]
            colors = vertex_colors[face_vertices]
            # Compute barycentric coordinates
            v0 = vertices[1] - vertices[0]
            v1 = vertices[2] - vertices[0]
            v2 = points[i] - vertices[0]
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            # Interpolate color using barycentric coordinates
            sampled_colors[i] = u * colors[0] + v * colors[1] + w * colors[2]
    except:
        # set color to gray
        sampled_colors = np.ones((num_points, 3)) * 0.5
    positions = points
    colors = sampled_colors
    normals = sampled_normals
    positions = np.array(positions)
    colors = np.array(colors) / 255.
    normals = np.array(normals)
    print("positions", positions.shape, "colors", colors.shape, "normals", normals.shape)
    storePly(mesh_path.replace('point_cloud.obj', 'points3d_gt.ply'), positions, colors * 255)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


if __name__ == "__main__":
    FLAGS = parse_args()
    blender_location = FLAGS.blender_location
    res = FLAGS.res
    views = FLAGS.views
    distance = FLAGS.distance
    obj_save_dir = FLAGS.obj_save_dir
    mode = FLAGS.mode
	
    if mode == 0:
        scale_obj(blender_location, obj_save_dir)
    elif mode == 1:
        total_setting = len(res) * len(views) * len(distance)
        cur_setting = 1
        for r in res:
            for v in views:
                for d in distance:
                    print('Start rendering: Param Setting %d / %d' % (cur_setting, total_setting))
                    render_obj(blender_location, obj_save_dir, r, v, d)
                    cur_setting += 1
    elif mode == 2:
        render_ref(blender_location, obj_save_dir)