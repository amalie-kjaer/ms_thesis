import h5py
import json
import pandas as pd
import os
import numpy as np
import glob
import io
from configparser import ConfigParser
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import wandb
from PIL import Image

def load_config(config_path='./config.yaml'):
    config = ConfigParser()
    config.read(config_path)
    return config

def construct_M(height_pixels, width_pixels):
    fov_x = np.pi/3.0
    fov_y = 2.0 * np.arctan(height_pixels * np.tan(fov_x/2.0) / width_pixels)
    near  = 1.0
    far   = 1000.0

    f_h    = np.tan(fov_y/2.0)*near
    f_w    = f_h*width_pixels/height_pixels
    left   = -f_w
    right  = f_w
    bottom = -f_h
    top    = f_h

    M_proj      = np.matrix(np.zeros((4,4)))
    M_proj[0,0] = (2.0*near)/(right - left)
    M_proj[1,1] = (2.0*near)/(top - bottom)
    M_proj[0,2] = (right + left)/(right - left)
    M_proj[1,2] = (top + bottom)/(top - bottom)
    M_proj[2,2] = -(far + near)/(far - near)
    M_proj[3,2] = -1.0
    M_proj[2,3] = -(2.0*far*near)/(far - near)
    
    return M_proj

def T_world2screen(p_world, camera_pos, camera_rot, height_pixels, width_pixels):
    R_cam2world = np.matrix(camera_rot)
    t_cam2world = np.matrix(camera_pos).T
    R_world2cam = R_cam2world.T
    t_world2cam = -R_world2cam*t_cam2world

    M = construct_M(height_pixels, width_pixels)

    p_cam      = t_world2cam + R_world2cam*p_world
    p_cam_     = np.matrix(np.r_[ p_cam.A1, 1 ]).T
    p_clip     = M * p_cam_
    p_ndc      = p_clip/p_clip[3]
    p_ndc_     = p_ndc.A1
    p_screen_x = 0.5*(p_ndc_[0]+1)*(width_pixels-1)
    p_screen_y = (1 - 0.5*(p_ndc_[1]+1))*(height_pixels-1)
    p_screen_z = (p_ndc_[2]+1)/2.0
    p_screen   = np.matrix([p_screen_x, p_screen_y, p_screen_z]).T
    
    return p_screen

def T_obj2world(p_obj, t_obj2world, R_obj2world):
    p_world = t_obj2world + R_obj2world*p_obj
    return p_world

def T_obj2screen(p_obj, cam_pos, cam_rot, t_obj2world, R_obj2world, height_pixels, width_pixels):
    p_world = T_obj2world(p_obj, t_obj2world, R_obj2world)
    p_screen = T_world2screen(p_world, cam_pos, cam_rot, height_pixels, width_pixels)
    return p_screen