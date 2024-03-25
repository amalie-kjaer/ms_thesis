from pathlib import Path
import os
import matplotlib.image as mpimg
import h5py
import numpy as np
import torch
import clip
from PIL import Image
import argparse
import json
from operator import add

# TODO: Add top-k frame-views for each instance to the scene_dict
# TODO: Add camera position and rotation to load_frame_data
# TODO: Make each scene a class (with n frames), with methods to load scene data, load frame data, process data, and visualize data

def load_scene_data(download_dir, scene_name, cam_name):
    # Load all scene-specific data
    bb_x_world_path = os.path.join(
        download_dir,
        scene_name,
        "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5"
        )
    with h5py.File(bb_x_world_path, "r") as f: bb_x_world = f['dataset'][:]

    bb_extents_world_path = os.path.join(
        download_dir,
        scene_name,
        "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5"
        )
    with h5py.File(bb_extents_world_path, "r") as f: bb_extents_world = f['dataset'][:]

    bb_rot_world_path = os.path.join(
        download_dir,
        scene_name,
        "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5")
    with h5py.File(bb_rot_world_path, "r") as f: bb_rot_world = f['dataset'][:]
    
    scene_data = {
        'download_dir': download_dir,
        'scene_name': scene_name,
        'cam_name': cam_name,
        'bb_x_world': bb_x_world,
        'bb_extents_world': bb_extents_world,
        'bb_rot_world': bb_rot_world
        }
    
    return scene_data

def load_frame_data(scene_data, frame_idx):
    # Load tonemap for the frame
    tonemap_dir = os.path.join(
        scene_data['download_dir'],
        scene_data['scene_name'],
        "images",
        "scene_" + scene_data['cam_name'] + "_final_preview",
        "frame." + frame_idx + ".tonemap.jpg"
        )
    tonemap = mpimg.imread(tonemap_dir)

    # Load semantic nyu segmentation for the frame
    sem_nyu_path = os.path.join(
        scene_data['download_dir'],
        scene_data['scene_name'],
        "images",
        "scene_" + scene_data['cam_name'] + "_geometry_hdf5",
        "frame." + frame_idx + ".semantic.hdf5"
        )
    with h5py.File(sem_nyu_path, "r") as f: 
        semantic_nyu = f['dataset'][:]
    
    # Load instance segmentation for the frame
    ins_seg_path = os.path.join(
        scene_data['download_dir'],
        scene_data['scene_name'],
        "images",
        "scene_" + scene_data['cam_name'] + "_geometry_hdf5",
        "frame." + frame_idx + ".semantic_instance.hdf5"
        )
    with h5py.File(ins_seg_path, "r") as f:
        ins_seg = f['dataset'][:]

    # Load 3d coordinates of each pixel for the frame (used to construct point cloud of the scene)
    X_3d_path = os.path.join(
        scene_data['download_dir'],
        scene_data['scene_name'],
        "images",
        "scene_" +  scene_data['cam_name'] + "_geometry_hdf5",
        "frame." + frame_idx + ".position.hdf5")
    with h5py.File(X_3d_path, "r") as f: X_3d = f['dataset'][:]
    
    camera_pos_dir = os.path.join(
        scene_data['download_dir'],
        scene_data['scene_name'],
        "_detail",
        scene_data['cam_name'],
        "camera_keyframe_positions.hdf5")
    with h5py.File(camera_pos_dir, "r") as f: camera_pos_all = f['dataset'][:]
    i = int(frame_idx.lstrip('0')) if frame_idx.lstrip('0') else 0
    cam_pos = camera_pos_all[i]

    camera_rot_dir = os.path.join(
        scene_data['download_dir'],
        scene_data['scene_name'],
        "_detail",
        scene_data['cam_name'],
        "camera_keyframe_orientations.hdf5")
    with h5py.File(camera_rot_dir, "r") as f: camera_rot_all = f['dataset'][:]
    cam_rot = camera_rot_all[i]

    frame_data = {
        'tonemap': tonemap,
        'semantic_nyu': semantic_nyu,
        'ins_seg': ins_seg,
        'cam_pos': cam_pos,
        'cam_rot': cam_rot,
        'X_3d': X_3d
    }

    return frame_data

def collect_instance_information(scene_data, frame_data, ins, new_instance=True):
    skip = False
    x_min = min(np.where(frame_data['ins_seg'] == ins)[0])
    x_max = max(np.where(frame_data['ins_seg'] == ins)[0])
    y_min = min(np.where(frame_data['ins_seg'] == ins)[1])
    y_max = max(np.where(frame_data['ins_seg'] == ins)[1])
    bb_extent = [int(x_min), int(y_min), int(x_max), int(y_max)]
    
    if x_max - x_min < 10 or y_max - y_min < 10: # skip small instances
        skip = True
        return None, skip
    else:
        image_features = get_clip_embedding(frame_data['tonemap'][x_min:x_max, y_min:y_max], type="image")

    if new_instance:
        class_id = frame_data['semantic_nyu'][frame_data['ins_seg']==ins][0]
        bb_centre_3d = scene_data['bb_x_world'][ins]
        instance_dict = {
            "instance_id": int(ins),
            "class_id": int(class_id),
            "bb_centre_3d": bb_centre_3d.tolist(),
            "clip_embedding": image_features[0].tolist(),
        }
        return instance_dict, skip
    else:
        return image_features, skip


def get_clip_embedding(x, type="image"):
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if type == "image":
        image = preprocess(Image.fromarray(x)).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
    elif type == "text":
        text = clip.tokenize(x).to(device)
        with torch.no_grad():
            features = model.encode_text(text)
    return features

def calculate_total_view_score(view):
    pass

def check_visibility(view):
    pass

def calculate_size_score(view):
    pass

def process_scene(scene_data):
    json_path = "scene_dict_new.json"
    if os.path.exists(json_path):
        scene_dict = json.load(open(json_path))
        print(f'Loaded scene_dict from {json_path}')
        
    else:
        # Initialize scene dictionary to store CLIP embeddings and other information of all instances in the scene
        scene_dict = {
            "scene_name": scene_data['scene_name'],
            "cam_name": scene_data['cam_name'],
            "instances": {}
        }
        ins_count = {}
        
        # Loop through frames in scene
        frames_in_scene = os.listdir(
            os.path.join(
                scene_data['download_dir'],
                scene_data['scene_name'],
                "images",
                "scene_" + scene_data['cam_name'] + "_final_preview"
                )
            )

        # TODO: Add another for loop here.
        # For each instance, calculate the top-k frame-views and add to scene_dict[instances][ins]["top_views"]
        # Then, calculate average CLIP embedding for each instance using the top-k frame-views (change code below accordingly)

        for f in frames_in_scene:
            
            frame_idx = f.split(".")[1]
            print(f'Processing frame {frame_idx}...')

            # Load frame data
            frame_data = load_frame_data(scene_data, frame_idx)

            # Loop through instances in frame
            ins_in_frame = np.unique(frame_data['ins_seg'])[1:] if np.unique(frame_data['ins_seg'])[0]==-1 else np.unique(frame_data['ins_seg'])
            for ins in ins_in_frame:
                ins = int(ins)

                # If first time seeing the instance, add all information to scene_dict
                if ins not in scene_dict["instances"]:
                    instance_dict, skip = collect_instance_information(scene_data, frame_data, ins, new_instance=True)
                    # Append instance information to scene_dict
                    if not skip:
                        scene_dict["instances"][ins] = instance_dict
                        ins_count[ins] = 1

                # If the instance is already in scene_dict (seen in previous frame), only add clip embedding to existing clip
                else:                    
                    image_features, skip = collect_instance_information(scene_data, frame_data, ins, new_instance=False)
                    if not skip:
                        scene_dict["instances"][ins]["clip_embedding"] = list(map(add, scene_dict["instances"][ins]["clip_embedding"], image_features[0].tolist()))
                        ins_count[ins] += 1

        # Average the clip embeddings
        for ins in scene_dict["instances"]:
            scene_dict["instances"][ins]["clip_embedding"][:] = [x / ins_count[ins] for x in scene_dict["instances"][ins]["clip_embedding"]]

        # Save scene_dict to json file
        json_file_path = "scene_dict_new.json"
        with open(json_file_path, "w") as json_file:
            json.dump(scene_dict, json_file, indent=4)

    return scene_dict

def visualize_instance_2d(scene_data, instance_id):
    frames_in_scene = os.listdir(
            os.path.join(
                scene_data['download_dir'],
                scene_data['scene_name'],
                "images",
                "scene_" + scene_data['cam_name'] + "_final_preview"
                )
            )
    for f in frames_in_scene:
        frame_idx = f.split(".")[1]
        frame_data = load_frame_data(scene_data, frame_idx)
        mylist=np.unique(frame_data['ins_seg'])
        if int(instance_id) in mylist:
            print(f'Visualizing instance {instance_id} in frame {frame_idx}...')
            x_min = min(np.where(frame_data['ins_seg'] == int(instance_id))[0])
            x_max = max(np.where(frame_data['ins_seg'] == int(instance_id))[0])
            y_min = min(np.where(frame_data['ins_seg'] == int(instance_id))[1])
            y_max = max(np.where(frame_data['ins_seg'] == int(instance_id))[1])

            img = frame_data['tonemap'][x_min:x_max, y_min:y_max]
            img = Image.fromarray(img)
            img.save('image_test.png')
            break
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="ai_001_001")
    parser.add_argument("--cam_name", type=str, default="cam_00")
    parser.add_argument("--download_dir", type=str, default="./contrib/99991/downloads/")
    parser.add_argument("--query", type=str, default="duck")
    # parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    return args

def main(args):
    scene_name = args.scene_name
    cam_name = args.cam_name
    download_dir=Path(args.download_dir)
    query = args.query

    # Load positions of instance bounding boxes in the scene
    scene_data = load_scene_data(download_dir, scene_name, cam_name)

    # Create a dictionary to store the scene data, with the CLIP embeddings of all object instances in the scene
    scene_dict = process_scene(scene_data)

    # Extract CLIP embeddings from scene_dict
    instance_clip = []
    for ins in scene_dict["instances"]:
        instance_clip.append(torch.tensor([scene_dict["instances"][ins]["clip_embedding"]]))
    instance_clip = torch.cat(instance_clip)

    # Get CLIP embedding for the query
    query_features = get_clip_embedding(query, type="text")   

    # Normalize the embeddings
    instance_clip /= instance_clip.norm(dim=0, keepdim=True)
    query_features /= query_features.norm(dim=0, keepdim=True)

    preds = (instance_clip @ query_features.T).softmax(dim=0)

    for i, k in enumerate(scene_dict["instances"].keys()):
        if i == torch.argmax(preds).cpu().detach().numpy():
            predicted_instance = k
            print(f'Predicted instance: {predicted_instance} with probability {torch.max(preds)}')

    visualize_instance_2d(scene_data, predicted_instance)
    


if __name__=="__main__":
    args = parse_args()    
    main(args)