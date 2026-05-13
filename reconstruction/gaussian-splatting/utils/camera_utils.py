#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
import zipfile
from io import BytesIO

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    # 检查图片路径是否在压缩包中
    if '.zip' in cam_info.image_path:
        # 分离压缩包路径和压缩包内的文件路径
        zip_path, internal_path = cam_info.image_path.split('.zip', 1)
        zip_path += '.zip'
        # 移除可能存在的前导斜杠
        internal_path = internal_path.lstrip('/')
        
        # 打开压缩包并读取图片
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            with zip_file.open(internal_path) as image_file:
                image_data = BytesIO(image_file.read())
                image = Image.open(image_data)
    else:
        # 检查路径中是否包含"image"文件夹，这可能是一个压缩包
        path_parts = cam_info.image_path.split('/')
        for i, part in enumerate(path_parts):
            if part == "image":
                # 构建可能的压缩包路径
                zip_path = '/'.join(path_parts[:i+1]) + '.zip'
                internal_path = '/'.join(path_parts[i+1:])
                
                try:
                    # 尝试打开压缩包
                    with zipfile.ZipFile(zip_path, 'r') as zip_file:
                        with zip_file.open(internal_path) as image_file:
                            image_data = BytesIO(image_file.read())
                            image = Image.open(image_data)
                            break
                except (FileNotFoundError, KeyError, zipfile.BadZipFile):
                    # 如果压缩包不存在或内部路径不正确，继续检查下一个可能的位置
                    continue
        else:
            # 如果没有找到匹配的压缩包，使用原始方式打开图片
            image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            # 检查深度图是否在压缩包中
            if '.zip' in cam_info.depth_path:
                # 分离压缩包路径和压缩包内的文件路径
                zip_path, internal_path = cam_info.depth_path.split('.zip', 1)
                zip_path += '.zip'
                # 移除可能存在的前导斜杠
                internal_path = internal_path.lstrip('/')
                
                # 打开压缩包并读取深度图
                with zipfile.ZipFile(zip_path, 'r') as zip_file:
                    with zip_file.open(internal_path) as depth_file:
                        depth_bytes = depth_file.read()
                        # 使用OpenCV从内存中读取深度图
                        depth_array = np.frombuffer(depth_bytes, np.uint8)
                        invdepthmap = cv2.imdecode(depth_array, -1)
                        if is_nerf_synthetic:
                            invdepthmap = invdepthmap.astype(np.float32) / 512
                        else:
                            invdepthmap = invdepthmap.astype(np.float32) / float(2**16)
            else:
                # 检查路径中是否包含"depth"文件夹，这可能是一个压缩包
                path_parts = cam_info.depth_path.split('/')
                depth_found = False
                for i, part in enumerate(path_parts):
                    if part in ["depth", "depths"]:
                        # 构建可能的压缩包路径
                        zip_path = '/'.join(path_parts[:i+1]) + '.zip'
                        internal_path = '/'.join(path_parts[i+1:])
                        
                        try:
                            # 尝试打开压缩包
                            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                                with zip_file.open(internal_path) as depth_file:
                                    depth_bytes = depth_file.read()
                                    # 使用OpenCV从内存中读取深度图
                                    depth_array = np.frombuffer(depth_bytes, np.uint8)
                                    invdepthmap = cv2.imdecode(depth_array, -1)
                                    if is_nerf_synthetic:
                                        invdepthmap = invdepthmap.astype(np.float32) / 512
                                    else:
                                        invdepthmap = invdepthmap.astype(np.float32) / float(2**16)
                                    depth_found = True
                                    break
                        except (FileNotFoundError, KeyError, zipfile.BadZipFile):
                            # 如果压缩包不存在或内部路径不正确，继续检查下一个可能的位置
                            continue
                
                if not depth_found:
                    # 原始方式读取深度图
                    if is_nerf_synthetic:
                        invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
                    else:
                        invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry