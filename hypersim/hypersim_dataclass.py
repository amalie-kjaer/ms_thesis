from dataclasses import dataclass, field
from dataclasses import asdict
from pathlib import Path
import os
import json
import matplotlib.image as mpimg
import h5py
import numpy as np
from utils_hypersim import T_obj2screen

@dataclass
class HypersimScene:
    scene: str
    camera: str
    instances: dict = field(init=False)
    # TODO: List other class attributes here (with init=False) for clarity
    download_dir: str = "C:/Users/amali/Documents/ds_research/ml-hypersim/contrib/99991/downloads/"
    
    def __post_init__(self):
        self.__check_scene_exists()
        
        ###########
        # TODO not sure these should be here, they are not strictly needed eg. for the visualisation
        ###########
        with h5py.File(os.path.join(self.download_dir, self.scene, "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5"), "r") as f:
            self.bb_centre_world = f['dataset'][:]
        with h5py.File(os.path.join(self.download_dir, self.scene, "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5"), "r") as f:
            self.bb_extents_world = f['dataset'][:]
        with h5py.File(os.path.join(self.download_dir, self.scene, "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5"), "r") as f:
            self.bb_rot_world = f['dataset'][:]
        
        # self.instances = self.__get_avg_instance_features()

    def __check_scene_exists(self):
        assert Path(self.download_dir).exists(), f"Download directory {self.download_dir} does not exist."
        assert Path(os.path.join(self.download_dir, self.scene)).exists(), f"Scene '{self.scene}' does not exist."
        assert Path(os.path.join(self.download_dir, self.scene, "_detail",self.camera)).exists(), f"Camera '{self.camera}' does not exist within scene '{self.scene}'."

    def get_avg_instance_features(self):
        instances = {}

        # for each instance, get score in each view (use get_frame_data['instance_scores'] OR __calculate_instance_visibility_score)
        # for each instance, select top 5 views
        # for each instance, calculate average clip embedding over top 5 views

        return instances

    def calculate_instance_visibility_score(self, frame_id: int, cam_pos, cam_rot, ins_seg):
        corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0]
        ]
        
        height_pixels, width_pixels = ins_seg.shape
        ins_in_frame = np.unique(ins_seg)[1:] if np.unique(ins_seg)[0]==-1 else np.unique(ins_seg)
        instance_scores = {}
        for ins in ins_in_frame:
            bb_extents = self.bb_extents_world[ins]
            bb_centre = self.bb_centre_world[ins]
            bb_rotation = self.bb_rot_world[ins]

            R_obj2world = np.matrix(bb_rotation)
            t_obj2world = np.matrix(bb_centre).T
            
            # Check how many pixels are visible
            pixels = len(np.where(ins_seg == ins)[0])

            # Check how many bounding box corners are visible
            corners_visible = 0
            for c in corners:
                corner_obj = np.diag(np.matrix(bb_extents).A1)*(np.matrix(c).T - 0.5)
                corner_screen = T_obj2screen(corner_obj, cam_pos, cam_rot, t_obj2world, R_obj2world, height_pixels, width_pixels)
                x = np.ravel(corner_screen[0]).astype(int).item()
                y = np.ravel(corner_screen[1]).astype(int).item()
                if x > 0 and x < width_pixels and y > 0 and y < height_pixels:
                    corners_visible +=1

            instance_scores[ins] = [pixels, corners_visible]
        return instance_scores

    def get_frame_data(self, frame_id: int):
        #########
        # TODO: This is quite inefficient because some application only need a small subset of files
        # (eg. build_pointcloud does not need to calculate instance_visibility_scores)
        #########

        tonemap = mpimg.imread(os.path.join(self.download_dir, self.scene, "images", "scene_" + self.camera + "_final_preview", "frame." + f"{frame_id:04}" + ".tonemap.jpg"))
        with h5py.File(os.path.join(self.download_dir, self.scene, "images", "scene_" + self.camera + "_geometry_hdf5", "frame." + f"{frame_id:04}" + ".semantic.hdf5"), "r") as f: 
            semantic_nyu = f['dataset'][:]
        with h5py.File(os.path.join(self.download_dir, self.scene, "images", "scene_" + self.camera + "_geometry_hdf5", "frame." + f"{frame_id:04}" + ".semantic_instance.hdf5"), "r") as f:
            ins_seg = f['dataset'][:]
        with h5py.File(os.path.join(self.download_dir, self.scene, "images", "scene_" + self.camera + "_geometry_hdf5", "frame." + f"{frame_id:04}" + ".position.hdf5"), "r") as f:
            X_3d = f['dataset'][:]
        with h5py.File(os.path.join(self.download_dir, self.scene, "_detail", self.camera, "camera_keyframe_positions.hdf5"), "r") as f:
            camera_pos_all = f['dataset'][:]
        cam_pos = camera_pos_all[frame_id]
        with h5py.File(os.path.join(self.download_dir, self.scene, "_detail", self.camera, "camera_keyframe_orientations.hdf5"), "r") as f:
            camera_rot_all = f['dataset'][:]
        cam_rot = camera_rot_all[frame_id]
        instance_scores = self.__calculate_instance_visibility_score(frame_id, cam_pos, cam_rot, ins_seg)

        frame_data = {
            'tonemap': tonemap,
            'nyu_semantic_segmentation': semantic_nyu,
            'instance_segmentation': ins_seg,
            'camera_position': cam_pos,
            'camera_rotation': cam_rot,
            'pixels_in_world_coordinates': X_3d,
            'instance_scores': instance_scores
        }
        return frame_data

    def build_pointcloud():
        None

    def plot_pointcloud():
        None
    
    def compare_query(self, query: str):
        None


scene_info = HypersimScene(scene='ai_001_001', camera='cam_00')

# print(scene_info.download_dir)
print(scene_info.scene)
# print(scene_info.bb_centre_world)
# print(scene_info.instances)
print(scene_info.get_frame_data(0)['instance_scores'])