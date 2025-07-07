from __init__ import *

import os
import json
import pickle
import argparse
import copy
import torch
from pathlib import Path
import open3d as o3d
import numpy as np

from utils import makeTpose, rotation_6d_to_matrix, BodyMaker, HandMaker, \
                    simpleViewer, get_stickman, get_stickhand,\
                    ContactViewer, compute_sdf, visualize_contact, get_textannot, get_annotation, annot2item

SCAN_ROOT = os.path.join(ROOT_REPOSITORY, "data/scan")

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", default="data/seq/s1", help="target path, e.g : ")
    parser.add_argument("--start_frame", default=0, type=int, help="Render start frame number") 
    parser.add_argument("--end_frame", default=100000, type=int, help="Render end frame number")
    parser.add_argument("--run", default=False, action='store_true', help='If set, viewer will show start_frame scene at specific view')
    parser.add_argument("--fromort", default=False, action='store_true', help='Make skeleton from orientation, transform to ')
    parser.add_argument("--ego", action='store_true')
    parser.add_argument("--smplx", action='store_true')
    parser.add_argument("--contact_tgs",type=str,nargs='+', default=[])

    args = parser.parse_args()
    
    root = os.path.join(ROOT_REPOSITORY, args.scene_root)
    camera_dir = root + "/cam_param"

    annot_path = root+"/text_annotation.json"
    text_annot = get_textannot(annot_path)    

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
        
        if args.ego:
            body_global_trans = pickle.load(open(root + "/body_global_transform.pkl", "rb"))
            body_ort = pickle.load(open(root + "/body_joint_orientations.pkl", "rb"))
            body_rotmat = rotation_6d_to_matrix(torch.tensor(body_ort)).numpy()   


    if args.smplx:
        import smplx
        smplroot =  root.replace('seq','smplx_seq')
        SMPLX_MODEL_PATH = '/home/jisoo/data2/git_repo/smplx/transfer_data/models'

        smplx_params = pickle.load(open(smplroot + "/smplx_params.pkl", "rb"))
        smplx_pose = pickle.load(open(smplroot + "/smplx_pose.pkl", "rb"))

        smplx_beta, gender = smplx_params['beta'].to(device), smplx_params['gender']
        body_pose = smplx_pose['body_pose'].reshape((-1,21,3)).to(device)
        global_orient = smplx_pose['global_orient'].to(device)
        transl = smplx_pose['transl'].to(device)
        hand_pose = smplx_pose['hand_pose'].reshape((-1,30,3)).to(device)
        lhand_pose = hand_pose[:,:15,:].to(device)
        rhand_pose = hand_pose[:,15:,:].to(device)

        body_model = smplx.create(model_path = SMPLX_MODEL_PATH,
                        model_type = "smplx",
                        flat_hand_mean=True,
                        use_pca=False,
                        num_betas = 20,
                        num_expression_coeffs = 10,
                        gender=gender,
                        ext='pkl').to(device)

        smplx_faces = body_model.faces


    frame_length = joint_rgb.shape[0]

    os.path.dirname(__file__)
    obj_color = json.load(open(os.path.join(ROOT_REPOSITORY,"visualize/color.json"), "r"))
    object_transform = pickle.load(open(root + "/object_transformations.pkl", "rb"))

    
    print("------Reading Objects------")
    initialized, mesh_dict, mesh_dict_copied = dict(), dict(), dict()
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
                mesh_dict_copied[keyn] = copy.deepcopy(m)


    # make camera parameters
    width, height = 1600, 800
    if args.ego:
        view = o3d.camera.PinholeCameraParameters()
        camera_matrix = np.eye(3, dtype=np.float64)
        f = 520
        camera_matrix[0,0] = f
        camera_matrix[1,1] = f
        camera_matrix[0,2] = width/2
        camera_matrix[1,2] = height/2
        view.intrinsic.intrinsic_matrix = camera_matrix
        view.intrinsic.width, view.intrinsic.height = width, height
    else:
        view=None

    vis = simpleViewer("Render Scene", 1600, 800, [], view) # 

    mesh_viewer_dict = {}
    if len(args.contact_tgs)>0:
        if args.smplx:
            output_glb = body_model()
            # Extract T-Posed Body Mesh
            verts_glb = output_glb.vertices[0]#*body_scale
            jts_glb = output_glb.joints[0]

            bmesh = o3d.geometry.TriangleMesh()
            bmesh_vertices = verts_glb.detach().cpu().numpy()
            bmesh.vertices = o3d.utility.Vector3dVector(bmesh_vertices)
            bmesh.triangles = o3d.utility.Vector3iVector(smplx_faces)
            bmesh.compute_vertex_normals()
            bmesh.paint_uniform_color([0.4,0.4,0.4])
            
            mesh_viewer_dict['body'] = ContactViewer('body',512,512,bmesh)
            mesh_viewer_dict['body_backward'] = ContactViewer('body_backward',512,512,bmesh)
            mesh_viewer_dict['lhand'] = ContactViewer('lhand',512,512,bmesh)
            mesh_viewer_dict['rhand'] = ContactViewer('rhand',512,512,bmesh)

            for obj_nm in args.contact_tgs:
                mesh_viewer_dict[obj_nm] = ContactViewer(obj_nm,512,512,mesh_dict[f'{obj_nm}_base'])

        else:
            print("You should use smplx rendering to compute contact")

    global_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
    vis.add_geometry({"name":"global", "geometry":global_coord})    
    vis.add_plane()
    for fn in range(args.start_frame, min(args.end_frame+1, frame_length)):
        """
        Head tip position 
        """
        if fn not in object_transform:
            continue

        cur_human_joints = joint_rgb[fn]
        
        cur_object_pose = object_transform[fn]

        if args.smplx:
            output_glb = body_model(betas=smplx_beta,
                    return_verts=True,
                    body_pose = body_pose[fn:fn+1], # 1X21X3
                    left_hand_pose = lhand_pose[fn:fn+1], #1X15X3
                    right_hand_pose = rhand_pose[fn:fn+1],#1X15X3
                    global_orient = global_orient[fn:fn+1], #1X3
                    transl = transl[fn:fn+1] #+ tl,
                    )
            verts_glb = output_glb.vertices[0]#*body_scale
            jts_glb = output_glb.joints[0]

            bmesh = o3d.geometry.TriangleMesh()
            bmesh_vertices = verts_glb.detach().cpu().numpy()
            bmesh.vertices = o3d.utility.Vector3dVector(bmesh_vertices)
            bmesh.triangles = o3d.utility.Vector3iVector(smplx_faces)
            bmesh.compute_vertex_normals()
            bmesh.paint_uniform_color([0.4,0.4,0.4])

            vis.add_geometry({"name":"human", "geometry":bmesh})
        else:
            bmesh = get_stickman(cur_human_joints[:23], head_tip_position[fn] if not args.ego else None)
            hmesh = get_stickhand(cur_human_joints[23:48]) + get_stickhand(cur_human_joints[48:])# hand
            bmesh.compute_vertex_normals()
            hmesh.compute_vertex_normals()

            vis.add_geometry({"name":"human", "geometry":bmesh+hmesh})


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


        annot = get_annotation(text_annot, fn)
        if annot is not None:
            interacting_objects = annot2item[annot]
        else:
            interacting_objects = []

        # Compute & Visualize Contact
        body_contacts = []
        for obj_nm in args.contact_tgs:
            if obj_nm in interacting_objects:
                # compute SDF using body mesh
                if f'{obj_nm}_base' in cur_object_pose:
                    transferred_objmesh = copy.deepcopy(mesh_dict_copied[f'{obj_nm}_base']).transform(cur_object_pose[f'{obj_nm}_base'])
                    body_contact = compute_sdf(bmesh, transferred_objmesh) # bmesh's contact to transferred objmesh
                    obj_contact = compute_sdf(transferred_objmesh, bmesh)
                    obj_color =  visualize_contact(obj_contact)
                    mesh_viewer_dict[obj_nm].update_color(obj_color)
                    body_contacts.append(body_contact)

        # Get Minimum of body
        if len(body_contacts)>0:
            # print("combine body ")
            body_contact = np.min(np.stack(body_contacts), axis=0)
            body_color =  visualize_contact(body_contact)
            for body_component in ['body','body_backward','lhand','rhand']:
                mesh_viewer_dict[body_component].update_color(body_color)

                
        if args.ego:
            head_posinrgb = cur_human_joints[5]
            head_rotinrgb = body_global_trans[fn,:3,:3]@body_rotmat[fn, 6]
            # Change Axis Directions 
            head_rotinrgb = np.stack([-head_rotinrgb[:,1],-head_rotinrgb[:,2],head_rotinrgb[:,0]],axis=1)
            head_Trgb = np.eye(4, dtype=np.float64)
            head_Trgb[:3,:3] = head_rotinrgb
            head_Trgb[:3,3] = head_posinrgb
            extrinsic_matrix = np.linalg.inv(head_Trgb)
            vis.setupcamera(extrinsic_matrix)

        if args.run:
            vis.run()        
        else:
            vis.tick()
        vis.remove_geometry("human")




        


    
