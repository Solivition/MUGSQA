import shutil
import math
from math import radians
import bpy
import numpy as np
import argparse
import sys
import os
import json
import copy
import cv2
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
    description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('obj', type=str,
                        help='Path to the obj file to be rendered.')
    parser.add_argument('--res', type=int, default=1080,
                        help='resolution of input images')
    parser.add_argument('--views', type=int, default=72,
                        help='number of views to be rendered')
    parser.add_argument('--distance', type=float, default=2,
                        help='Distance between camera and model.')
    parser.add_argument('--remove_doubles', type=bool, default=False,
                        help='Remove double vertices to improve mesh quality.')
    parser.add_argument('--edge_split', type=bool, default=False,
                        help='Adds edge split filter.')
    parser.add_argument('--depth_scale', type=float, default=1.4,
                        help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
    parser.add_argument('--color_depth', type=str, default='8',
                        help='Number of bit per channel used for output. Either 8 or 16.')
    parser.add_argument('--format', type=str, default='PNG',
                        help='Format of files generated. Either PNG or OPEN_EXR.')
    parser.add_argument('--obj_save_dir', type=str,
                        help='Path to the render file to be saved.')
    parser.add_argument('--origin_center', type=str, default='MEDIAN',
                        help='BOUNDS or MEDIAN.')
    parser.add_argument('--ref_mode', type=bool, default=False,
                        help='True: Render imgs, False: Render videos')
    argv = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(argv)


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


# Render settings
def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty
    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    return b_empty


def camera_info(param):
    "params: [theta, phi, rho, x, y, z, f]"
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])
    camY = param[3]*np.sin(phi) * param[6]
    temp = param[3]*np.cos(phi) * param[6]
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    print("cam axis", camX, camY, camZ)
    return camX, -camZ, camY


def render_input(views, obj_save_dir):
    bpy.context.scene.render.film_transparent = True
    stepsize = 360.0 / (views)
    stepsize_v = 72//views
    # manual vertical
    vertical_list = [-np.pi/4, 0, np.pi/5] * (72//3)
    vertical_list = [-vertical_list[i] if i//3%2 else vertical_list[i] for i in range(len(vertical_list))]
    out_data['frames'] = []
    b_empty.rotation_euler = (0, 0, 0)
    b_empty.rotation_euler[0] = vertical_list[0]
    for i in range(views):
        if i != 0:
            b_empty.rotation_euler[0] = vertical_list[i*stepsize_v]
            b_empty.rotation_euler[2] += radians(stepsize)
        scene.render.filepath = obj_save_dir + '/image/' + str(i).zfill(3)
        tree.nodes['Depth Output'].file_slots[0].path = "/depth/" + str(i).zfill(3)
        tree.nodes['Normal Output'].file_slots[0].path = "/normal/" + \
        str(i).zfill(3)
        bpy.ops.render.render(write_still=True)  # render still
        frame_data = {
            'file_path': 'image/' + str(i).zfill(3),
            'rotation': radians(stepsize),
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        out_data['frames'].append(frame_data)
    with open(obj_save_dir + '/' + 'transforms_train.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)
    test_json = copy.deepcopy(out_data)
    test_json['frames'] = test_json['frames'][-4:]
    with open(os.path.join(obj_save_dir, 'transforms_test.json'), 'w') as f:
        json.dump(test_json, f, indent=4)
    with open(os.path.join(obj_save_dir, 'transforms_val.json'), 'w') as f:
        json.dump(test_json, f, indent=4)
    # zip image file, depth files and normals
    print("zip image file, depth files and normals")
    shutil.make_archive(os.path.join(obj_save_dir, 'image'),
                        'zip', os.path.join(obj_save_dir, 'image'))
    shutil.make_archive(os.path.join(obj_save_dir, 'depth'),
                        'zip', os.path.join(obj_save_dir, 'depth'))
    shutil.make_archive(os.path.join(obj_save_dir, 'normal'),
                        'zip', os.path.join(obj_save_dir, 'normal'))
    shutil.rmtree(os.path.join(obj_save_dir, 'image'))
    shutil.rmtree(os.path.join(obj_save_dir, 'depth'))
    shutil.rmtree(os.path.join(obj_save_dir, 'normal'))


def render_ref(obj_save_dir):
    bpy.context.scene.render.film_transparent = True
    stepsize = 6.0
    b_empty.rotation_euler = (0, 0, 0)
    distance_list = [1.8 - 0.01*i for i in range(30)] + [1.5 - 0.01*i for i in range(30)] + [1.2 for i in range(30)] + [1.2 + 0.01*i for i in range(30)] + [1.5 + 0.01*i for i in range(30)] + [1.8 for i in range(30)]
    out_data['frames'] = []
    # Rotation Part I
    for i in range(180):
        cam.location = (0, -distance_list[i], 0)
        if i != 0:
            b_empty.rotation_euler[2] += radians(stepsize)
        scene.render.filepath = obj_save_dir + '/image/' + str(i).zfill(3)
        tree.nodes['Depth Output'].file_slots[0].path = "/depth/" + str(i).zfill(3)
        tree.nodes['Normal Output'].file_slots[0].path = "/normal/" + \
        str(i).zfill(3)
        bpy.ops.render.render(write_still=True)  # render still
        frame_data = {
            'file_path': 'image/' + str(i).zfill(3),
            'rotation': radians(stepsize),
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        out_data['frames'].append(frame_data)

    # Generate video
    fps = 30
    output_video_path = os.path.join(obj_save_dir, 'GT_video.mp4')
    img_path = obj_save_dir + '/image/'
    # Get a list of image file names and sort them
    images = sorted([img for img in os.listdir(img_path) if img.endswith(("png", "jpg", "jpeg"))])
    # Read the first image to determine the frame size
    first_image_path = os.path.join(img_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    size = (width, height)
    # Creating a Video Write Object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using the MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    # Write each image to the video in order
    video_path = obj_save_dir + '/video_image/'
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    for image in images:
        cur_img_path = os.path.join(img_path, image)
        video_img_path = os.path.join(video_path, image)
        image = Image.open(cur_img_path).convert("RGBA")
        background = Image.new("RGB", image.size, (153, 153, 153))
        background.paste(image, (0, 0), image)
        background.save(video_img_path)
        frame = cv2.imread(video_img_path)
        if frame is None:
            print(f"Failed to read image: {cur_img_path}")
            continue
        video_writer.write(frame)
    # Release of resources
    video_writer.release()
    print(f"Video saved to {output_video_path}")

    with open(obj_save_dir + '/' + 'transforms_train.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)
    test_json = copy.deepcopy(out_data)
    test_json['frames'] = test_json['frames'][-4:]
    with open(os.path.join(obj_save_dir, 'transforms_test.json'), 'w') as f:
        json.dump(test_json, f, indent=4)
    with open(os.path.join(obj_save_dir, 'transforms_val.json'), 'w') as f:
        json.dump(test_json, f, indent=4)
    # zip image file, depth files and normals
    print("zip image file, depth files and normals")
    shutil.make_archive(os.path.join(obj_save_dir, 'image'),
                        'zip', os.path.join(obj_save_dir, 'image'))
    shutil.make_archive(os.path.join(obj_save_dir, 'depth'),
                        'zip', os.path.join(obj_save_dir, 'depth'))
    shutil.make_archive(os.path.join(obj_save_dir, 'normal'),
                        'zip', os.path.join(obj_save_dir, 'normal'))
    shutil.rmtree(os.path.join(obj_save_dir, 'image'))
    shutil.rmtree(os.path.join(obj_save_dir, 'depth'))
    shutil.rmtree(os.path.join(obj_save_dir, 'normal'))


if __name__ == "__main__":
    args = parse_args()

    print("start rendering")

    # Set up rendering of depth map using nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_combined = True
    bpy.context.view_layer.use_pass_z = True
    bpy.context.scene.render.image_settings.file_format = args.format
    bpy.context.scene.render.image_settings.color_depth = args.color_depth

    render_layers_node = tree.nodes.new('CompositorNodeRLayers')
    # Setup for output of depth
    # Link nodes
    links = tree.links

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.base_path = args.obj_save_dir
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    # Remap as other types can not represent the full range of depth.
    map = tree.nodes.new(type="CompositorNodeMapRange")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.inputs['From Min'].default_value = 0
    map.inputs['From Max'].default_value = 8
    map.inputs['To Min'].default_value = 1
    map.inputs['To Max'].default_value = 0
    links.new(render_layers_node.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

    #  Setup for output of normals
    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.base_path = args.obj_save_dir
    normal_file_output.label = 'Normal Output'
    normal_file_output.name = 'Normal Output'
    links.new(render_layers_node.outputs['Normal'], normal_file_output.inputs[0])

    # Import OBJ file
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    imported_obj = bpy.ops.import_scene.obj(filepath=args.obj)
    # Assumes imported obj has one main object
    obj_object = bpy.context.selected_objects[0]
    # Reset object's location
    obj_object.location = (0, 0, 0)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center=args.origin_center)
    # Scale, remove doubles, add edge split if specified
    if args.remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if args.edge_split:
        modifier = obj_object.modifiers.new(name='EdgeSplit', type='EDGE_SPLIT')
        modifier.split_angle = 1.32645
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Setup resolution
    scene = bpy.context.scene
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100

    # Background
    bpy.context.scene.render.dither_intensity = 0.0
    # White background with lighting
    bpy.context.scene.render.film_transparent = False
    bg_nodes = bpy.context.scene.world.node_tree.nodes
    bg_nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

    # Data to store in JSON file
    out_data = {
        'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    }

    # Setup camera
    cam = scene.objects['Camera']
    cam.location = (0, -args.distance, 0)
    b_empty = parent_obj_to_camera(cam)

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_constraint.target = b_empty

    scene.render.image_settings.file_format = args.format

    views = args.views
    obj_save_dir = args.obj_save_dir

    if args.ref_mode:
        render_ref(obj_save_dir)
    else:
        render_input(views, obj_save_dir)