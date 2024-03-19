from pathlib import Path
import os
import matplotlib.image as mpimg
import h5py
import numpy as np
import open3d as o3d
import argparse
from extract_clip import load_scene_data, load_frame_data

def build_colormap(scene_data, color_instances=False, color_semantic=False):
    if color_instances:
        n_colors = scene_data['ins_bb_pos'].shape[0]
    elif color_semantic:
        n_colors = 40
    seg_ids = np.arange(1, n_colors + 1)
    colors = np.random.randint(0, 256, size=(n_colors, 3))
    colormap = dict(zip(seg_ids, colors))
    colormap[-1] = np.random.randint(0, 256, size=(1,3))
    return colormap

def map_instance_to_color(seg, colormap):
    # np.random.seed(0)
    seg_ids = np.unique(seg)

    h, w = seg.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for i in seg_ids:
        mask = seg == i
        rgb_image[mask] = colormap[i]

    return rgb_image

def remove_top():
    # Remove top 10% of points
    pass

def build_pointcloud(scene_data, args):
    color_instances = args.color_instances
    color_semantic = args.color_semantic
    
    point_cloud_path = "ai_001_001.ply"
    point_cloud_instances_path = "ai_001_001_instances.ply"
    point_cloud_semantic_path = "ai_001_001_semantic.ply"
    
    # Load pointcloud if it already exists
    if os.path.exists(point_cloud_path) and color_instances == False and color_semantic == False:
        print('Reading point cloud with regular colours...')
        pcd_load = o3d.io.read_point_cloud(point_cloud_path)
    elif os.path.exists(point_cloud_instances_path) and color_instances:
        print('Reading point cloud with instance colours...')
        pcd_load = o3d.io.read_point_cloud(point_cloud_instances_path)
    elif os.path.exists(point_cloud_semantic_path) and color_semantic:
        print('Reading point cloud with semantic colours...')
        pcd_load = o3d.io.read_point_cloud(point_cloud_semantic_path)
    
    # If pointcloud does not exist, construct it
    else:
        frames_in_scene = os.listdir(
        os.path.join(
            scene_data['download_dir'],
            scene_data['scene_name'],
            "images",
            "scene_" + scene_data['cam_name'] + "_final_preview"
            )
        )
        for i, f in enumerate(frames_in_scene):
            frame_idx = f.split(".")[1]
            print(f'Processing frame {frame_idx}...')
            frame_data = load_frame_data(scene_data, frame_idx)
            if i == 0: # Instantiate point cloud for the first frame
                X_3d = frame_data['X_3d'].reshape(-1, 3)
                if color_instances==False and color_semantic==False:
                    rgb_image = (frame_data['tonemap']/255).reshape(-1, 3)
                    path = point_cloud_path
                elif color_instances==True:
                    ins_seg = frame_data['ins_seg']
                    colormap = build_colormap(scene_data, color_instances=True)
                    rgb_image = map_instance_to_color(ins_seg, colormap)
                    rgb_image = (rgb_image/255).reshape(-1, 3)
                    path = point_cloud_instances_path
                elif color_semantic==True:
                    sem_seg = frame_data['semantic_nyu']
                    colormap = build_colormap(scene_data, color_semantic=True)
                    rgb_image = map_instance_to_color(sem_seg, colormap)
                    rgb_image = (rgb_image/255).reshape(-1, 3)
                    path = point_cloud_semantic_path
            else: # Concatenate points and colors for subsequent frames
                X_3d = np.concatenate((X_3d, frame_data['X_3d'].reshape(-1, 3)), axis=0)
                if color_instances==False and color_semantic==False:
                    rgb_image = np.concatenate((rgb_image, (frame_data['tonemap']/255).reshape(-1, 3)), axis=0)
                elif color_instances==True:
                    ins_seg = frame_data['ins_seg']
                    temp = map_instance_to_color(ins_seg, colormap)
                    temp = (temp/255).reshape(-1, 3)
                    rgb_image = np.concatenate((rgb_image, temp), axis=0)
                elif color_semantic==True:
                    sem_seg = frame_data['semantic_nyu']
                    temp = map_instance_to_color(sem_seg, colormap)
                    temp = (temp/255).reshape(-1, 3)
                    rgb_image = np.concatenate((rgb_image, temp), axis=0)

        print('Writing point cloud...')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X_3d)
        pcd.colors = o3d.utility.Vector3dVector(rgb_image)
        o3d.io.write_point_cloud(path, pcd)
    
        print('Reading point cloud...')
        pcd_load = o3d.io.read_point_cloud(path)
    
    print('Visualizing point cloud...')
    o3d.visualization.draw_geometries([pcd_load])
    print('done')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="ai_001_001")
    parser.add_argument("--cam_name", type=str, default="cam_00")
    parser.add_argument("--download_dir", type=str, default="./contrib/99991/downloads/")
    parser.add_argument("--color_instances", action="store_true")
    parser.add_argument("--color_semantic", action="store_true")
    parser.add_argument("--light_version", action="store_true")
    parser.add_argument("--remove_top", action="store_true")

    args = parser.parse_args()
    return args

def main(args):
    scene_name = args.scene_name
    cam_name = args.cam_name
    download_dir=Path(args.download_dir)

    scene_data = load_scene_data(download_dir, scene_name, cam_name)
    
    build_pointcloud(scene_data, args)

if __name__=="__main__":
    args = parse_args()    
    main(args)