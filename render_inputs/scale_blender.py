import bpy
import os
import shutil
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_save_dir', type=str,
                        default='/path/to/datasets/MUGSQA_SOURCE',
                        help='Root directory containing one subfolder per source object.')
    parser.add_argument('--target_range', type=float, default=1.0)
    argv = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(argv)


def scale_obj_to_range(obj, target_range):
    # Get bounding box dimensions
    dimensions = obj.dimensions
    max_dimension = max(dimensions)
    # Calculate scale factor
    scale_factor = target_range / max_dimension
    # Apply scaling
    obj.scale = (scale_factor, scale_factor, scale_factor)


def process_folder(input_folder, target_range):
    for subfolder_name in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            # Output directory for subfolder
            output_subfolder = os.path.join(subfolder_path, 'calibrated')
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            subfolder_path = os.path.join(subfolder_path, 'source')
            # Iterate through files in subfolder
            for file_name in os.listdir(subfolder_path):
                # Copy the .jpg file to the output folder
                if file_name.endswith(".jpg"):
                    input_path = os.path.join(subfolder_path, file_name)
                    output_path = os.path.join(output_subfolder, file_name)
                    if os.path.exists(input_path):
                        shutil.copy(input_path, output_path)
                # Scale .obj files
                if file_name.endswith(".obj"):
                    input_path = os.path.join(subfolder_path, file_name)
                    output_path = os.path.join(output_subfolder, file_name)
                    # Copy the associated .mtl file to the output folder
                    mtl_file_name = os.path.splitext(file_name)[0] + ".mtl"
                    mtl_input_path = os.path.join(subfolder_path, mtl_file_name)
                    mtl_output_path = os.path.join(output_subfolder, mtl_file_name)
                    if os.path.exists(mtl_input_path):
                        shutil.copy(mtl_input_path, mtl_output_path)
                    # Check if scaled
                    if os.path.exists(output_path):
                        print("Exist!!! skip %s" % (output_subfolder))
                        continue
                    # Import model
                    bpy.ops.import_scene.obj(filepath=input_path)
                    obj = bpy.context.selected_objects[0]
                    # Scale model
                    scale_obj_to_range(obj, target_range)
                    # Export model, ensure texture path is correct
                    bpy.ops.export_scene.obj(filepath=output_path, use_selection=True, path_mode='RELATIVE')
                    # Delete imported model
                    bpy.ops.object.delete()


if __name__ == "__main__":
    args = parse_args()
    obj_save_dir = args.obj_save_dir
    target_range = args.target_range
    process_folder(obj_save_dir, target_range)