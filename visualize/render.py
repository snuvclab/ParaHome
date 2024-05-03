from __init__ import *

import os
import json
import pickle
import argparse
import torch
from pathlib import Path
import open3d as o3d
import numpy as np

from utils import makeTpose, rotation_6d_to_matrix, BodyMaker, HandMaker, \
                    simpleViewer, get_stickman, get_stickhand

SCAN_ROOT = os.path.join(ROOT_REPOSITORY, "data/scan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", default="data/seq/s01", help="target path, e.g : ")
    parser.add_argument("--start_frame", default=0, type=int, help="Render start frame number") 
    parser.add_argument("--end_frame", default=100000, type=int, help="Render end frame number")
    parser.add_argument("--run", default=False, action='store_true', help='If set, viewer will show start_frame scene at specific view')
    parser.add_argument("--fromort", default=False, action='store_true', help='Make skeleton from orientation, transform to ')

    args = parser.parse_args()
    
    root = os.path.join(ROOT_REPOSITORY, args.scene_root)
    camera_dir = scene_root + "/cam_param"

    
    head_tip_position = pickle.load(open(root + "/head_tips.pkl", "rb"))
    if args.fromort:
        bone_vector = pickle.load(open(root + "/bone_vectors.pkl", "rb"))
        bodyTpose, HandTpose = makeTpose(bone_vector)
        
        body_ort = pickle.load(open(root + "/body_joint_orientations.pkl", "rb"))
        hand_ort = pickle.load(open(root + "/hand_joint_orientations.pkl", "rb"))
        body_rotmat = rotation_6d_to_matrix(torch.tensor(body_ort)).numpy()
        hand_rotmat = rotation_6d_to_matrix(torch.tensor(hand_ort)).numpy()        

        bodymaker = BodyMaker(bodyTpose)
        handmaker = HandMaker(HandTpose)
        body_joint = bodymaker(torch.tensor(body_ort)).numpy()
        hand_joint = handmaker(torch.tensor(hand_ort)).numpy()

        lhand = np.einsum('FMN, FJN -> FJM', body_rotmat[:,14], hand_joint[:,0]) + body_joint[:, 14][:,None,:]
        rhand = np.einsum('FMN, FJN -> FJM', body_rotmat[:,10], hand_joint[:,1]) + body_joint[:, 10][:,None,:]        
        joint_root_fixed = np.concatenate([body_joint, lhand, rhand], axis=1)

        body_global_trans = pickle.load(open(root + "/body_global_transform.pkl", "rb"))
        joint_rgb = np.einsum('FMN, FJN->FJM',body_global_trans[:,:3,:3], joint_root_fixed) + body_global_trans[:, :3, 3][:,None,:]
    else:
        joint_rgb = pickle.load(open(root + "/joint_positions.pkl", "rb"))

    

    os.path.dirname(__file__)
    obj_color = json.load(open(os.path.join(ROOT_REPOSITORY,"visualize/color.json"), "r"))
    object_transform = pickle.load(open(root + "/object_transformations.pkl", "rb"))

    
    print("------Reading Objects------")
    initialized, mesh_dict = dict(), dict()
    obj_in_scene = json.load(open(Path(root, "object_in_scene.json"), "r"))
    # load_object
    for objn in obj_in_scene:
        for pn in ["base", "part1", "part2"]:
            meshpath = Path(SCAN_ROOT, objn, "simplified", pn+".obj")
            if meshpath.exists():
                keyn = objn + "_" + pn
                initialized[keyn] = False
                m = o3d.io.read_triangle_mesh(str(meshpath)) 
                m.paint_uniform_color(obj_color[objn][pn])
                m.compute_vertex_normals()
                mesh_dict[keyn] = m


    vis = simpleViewer("Render Scene", 1600, 800, [], None) # 

    global_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
    vis.add_geometry({"name":"global", "geometry":global_coord})    
    vis.add_plane()
    for fn in range(args.start_frame, args.end_frame+1):
        """
        Head tip position 
        """
        cur_human_joints = joint_rgb[fn]
        
        cur_object_pose = object_transform[fn]
        
        bmesh = get_stickman(cur_human_joints[:23], head_tip_position[fn])
        hmesh = get_stickhand(cur_human_joints[23:48]) + get_stickhand(cur_human_joints[48:])# hand
        bmesh.compute_vertex_normals()
        hmesh.compute_vertex_normals()
        
        vis.add_geometry({"name":"human", "geometry":bmesh+hmesh})

        # category 별로 나눠서 보기
        for inst_name, loaded in initialized.items():
            if loaded and inst_name in cur_object_pose:
                vis.transform(inst_name, cur_object_pose[inst_name])
            elif loaded and not inst_name in cur_object_pose: 
                vis.remove_geometry(inst_name)
                initialized[inst_name] = False
            elif not loaded and inst_name in cur_object_pose:
                vis.add_geometry({"name":inst_name, "geometry":mesh_dict[inst_name]})
                vis.transform(inst_name, cur_object_pose[inst_name])
                initialized[inst_name] = True
            elif not loaded and not inst_name in cur_object_pose:
                continue
        if fn == args.start_frame:
            vis.main_vis.reset_camera_to_default()

        if args.run:
            vis.run()        
        else:
            vis.tick()
        vis.remove_geometry("human")




        


    
