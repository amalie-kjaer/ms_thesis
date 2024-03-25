from dataclasses import dataclass, field
from dataclasses import asdict
from pathlib import Path
import os
import json

@dataclass
class HypersimScene:
    scene: str
    camera: str
    instances: dict = field(init=False)
    # instance_extents_world: list = field(init=False)
    # bb_x_world = field(init=False)
    # bb_rot_world = field(init=False)
    download_dir: str = "C:/Users/amali/Documents/ds_research/ml-hypersim/contrib/99991/downloads/"
    
    def __post_init__(self):
        self.__check_scene_exists()
        self.instances = self.__get_instance_features()

    def __check_scene_exists(self):
        assert Path(self.download_dir).exists(), f"Download directory {self.download_dir} does not exist."
        assert Path(os.path.join(self.download_dir, self.scene)).exists(), f"Scene '{self.scene}' does not exist."
        assert Path(os.path.join(self.download_dir, self.scene, "_detail",self.camera)).exists(), f"Camera '{self.camera}' does not exist within scene '{self.scene}'."

    def __get_instance_features(self):
        None 

    def frame_data(self, frame_id: int):
        None

    def build_pointcloud():
        None

    def plot_pointcloud():
        None


scene_info = HypersimScene(scene='ai_001_001', camera='cam_00')

print(scene_info.download_dir)
print(scene_info.scene)
print(scene_info.instances)