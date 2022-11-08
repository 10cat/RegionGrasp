import os
import sys
sys.path.append('.')
sys.path.append('..')
import pickle
from subprocess import Popen
import shutil # high-level file operations
import time
import tempfile
import numpy as np
import pybullet as p
import skvideo.io as skvio

from utils.utils import makepath

def take_picture(renderer, width=256, height=256, conn_id=None):
    view_matrix = p.computeViewMatrix([0, 0, -1], [0, 0, 0], [0, -1, 0],
                                      physicsClientId=conn_id)
    proj_matrix = p.computeProjectionMatrixFOV(20, 1, 0.05, 2, physicsClientId=conn_id)
    w, h, rgba, depth, mask = p.getCameraImage(width=width, height=height,
                                               projectionMatrix=proj_matrix,
                                               viewMatrix=view_matrix,
                                               renderer=renderer,
                                               physicsClientId=conn_id)

    return rgba

def write_video(frames, path):
    skvio.vwrite(path, np.array(frames).astype(np.uint8))

def process_sample(sample_idx, sample, save_gif_folder=None, save_obj_folder=None, vhacd_exe=None, use_gui=False, wait_time=0, sample_vis_freq=10, save_all_steps=True):
    if use_gui:
        conn_id = p.connect(p.GUI)
    else:
        conn_id = p.connect(p.DIRECT)

    if sample_idx % sample_vis_freq == 0:
        save_video = True
        save_video_path = os.path.join(save_gif_folder, "{:08d}.gif".format(sample_idx))
        save_obj_path = os.path.join(save_obj_folder, "{:08d}_obj.obj".format(sample_idx))
        save_hand_path = os.path.join(save_obj_folder, "{:08d}_hand.obj".format(sample_idx))
        if save_all_steps:
            save_obj_steps_folder = os.path.join(save_obj_folder, "{:08d}_obj".format(sample_idx))
            save_hand_steps_folder = os.path.join(save_obj_folder, "{:08d}_hand".format(sample_idx))
            makepath(save_obj_steps_folder)
            makepath(save_hand_steps_folder)
        makepath(save_video_path)
    else:
        save_video = False
        save_video_path = None
        save_obj_path = None
        save_hand_path = None
    # sample格式: (dict:{"hand_verts":(), "hand_faces":(), "obj_verts":(), "obj_faces":()})
    distance = run_simulation(hand_verts=sample["hand_verts"],
                              hand_faces=sample["hand_faces"],
                              obj_verts=sample["obj_verts"],
                              obj_faces=sample["obj_faces"],
                              conn_id=conn_id,
                              simulation_step=1 / 240,
                              object_friction=3, hand_friction=3,
                              hand_restitution=0, object_restitution=0.5,
                              object_mass=1, verbose=True, vhacd_resolution=1000, vhacd_exe=vhacd_exe,
                              wait_time=wait_time, save_video=save_video, save_obj_path=save_obj_path,
                              save_hand_path=save_hand_path, save_video_path=save_video_path, use_gui=use_gui)
    print("Distance = ", distance)
    return distance

def run_simulation(hand_verts, hand_faces, obj_verts, obj_faces,
                   conn_id=None, vhacd_exe=None, sample_idx=None,
                   save_video=False, save_video_path=None,
                   simulation_step=1 / 240, num_iterations=35,
                   object_friction=3, hand_friction=3, # friction parameter
                   hand_restitution=0, object_restitution=0.5,
                   object_mass=1, verbose=False, vhacd_resolution=1000, wait_time=0,
                   save_hand_path=None, save_obj_path=None,
                   save_simul_folder=None, use_gui=False):
    if conn_id is None:
        if use_gui:
            conn_id = p.connect(p.GUI)
        else:
            conn_id = p.connect(p.DIRECT)
    hand_indices = hand_faces.flatten().tolist() # hand_verts -- (list)
    p.resetSimulation(physicsClientId=conn_id)
    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=conn_id)
    p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=conn_id)
    p.setPhysicsEngineParameter(fixedTimeStep=simulation_step, physicsClientId=conn_id)
    p.setGravity(0, 9.8, 0, physicsClientId=conn_id) # => +y - axis is pointing vertical to the ground

    #---- add hand ----#
    base_tmp_dir = "tmp/objs"
    makepath(base_tmp_dir)
    hand_tmp_fname = tempfile.mktemp(suffix='.obj', dir=base_tmp_dir)
    save_obj(hand_tmp_fname, hand_verts, hand_faces)
    if save_hand_path is not None:
        shutil.copy(hand_tmp_fname, save_hand_path)
    hand_collision_id = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=hand_tmp_fname,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        indices=hand_indices,
        physicsClientId=conn_id) # collisionshape -> flag of collision

    hand_visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=hand_tmp_fname,
        rgbaColor=[0, 0, 1, 1],
        specularColor=[0, 0, 1],
        physicsClientId=conn_id) # visualshape -> color

    hand_body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=hand_collision_id,
        baseVisualShapeIndex=hand_visual_id,
        physicsClientId = conn_id) # hand_body -> visualshape (+) collisionshape
    
    p.changeDynamics(
        hand_body_id,
        -1,
        lateralFriction=hand_friction,
        restitution=hand_restitution,
        physicsClientId=conn_id)

    #TODO ---add obj----#
    obj_tmp_fname = tempfile.mktemp(suffix='.obj', dir=base_tmp_dir)
    makepath(base_tmp_dir)
    # Save object .obj file
    if save_obj_path is not None:
        final_obj_tmp_fname = tempfile.mktemp(suffix='.obj', dir=base_tmp_dir)
        save_obj(final_obj_tmp_fname, obj_verts, obj_faces)
        shutil.copy(final_obj_tmp_fname, save_obj_path)
    # Get obj center of mass
    obj_center_mass = np.mean(obj_verts, axis=0) #mean centroid
    obj_verts -= obj_center_mass # 平移 -> 转换至质心为原点的坐标系
    # add object
    use_vhacd = True
    if use_vhacd:
        # convex hull decomposition 
        if verbose:
            print("Computing vhacd decomposition")
            time1 = time.time()
        save_obj(obj_tmp_fname, obj_verts, obj_faces)
        # (Original: apply vhacd with vhacd.exe running )
        # if not vhacd(obj_tmp_fname, vhacd_exe, resolution=vhacd_resolution):
        #     raise RuntimeError(
        #         "Cannot compute convex hull "
        #         "decomposition for {}".format(obj_tmp_fname)
        #     )
        # else:
        #     print(f"Succeeded vhacd decomp of {obj_tmp_fname}")

        # TODO use built-in 'vhacd' function in pybullet
        name_log = "pybullet_vhacd.txt"
        p.vhacd(obj_tmp_fname, obj_tmp_fname, name_log)
        obj_collision_id = p.createCollisionShape(
            p.GEOM_MESH, fileName=obj_tmp_fname, physicsClientId=conn_id
        )
        if verbose:
            time2 = time.time()
            print(
                "Computed v-hacd decomposition at res {} {:.6f} s".format(
                    vhacd_resolution, (time2 - time1)
                )
            )
    else:
        obj_collision_id = p.createCollisionShape(
            p.GEOM_MESH, vertices=obj_verts, physicsClientId=conn_id
        )
    
    obj_visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=obj_tmp_fname,
        rgbaColor=[1, 0, 0, 1],
        specularColor=[1, 0, 0],
        physicsClientId=conn_id,
    )
    obj_body_id = p.createMultiBody(
        baseMass=object_mass,
        basePosition=obj_center_mass,
        baseCollisionShapeIndex=obj_collision_id,
        baseVisualShapeIndex=obj_visual_id,
        physicsClientId=conn_id,
    )

    p.changeDynamics(
        obj_body_id,
        -1,
        lateralFriction=object_friction,
        restitution=object_restitution,
        physicsClientId=conn_id,
    )

    # TODO --- simulate for several steps --- #
    if save_video:
        images = []
        if use_gui:
            renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            renderer = p.ER_TINY_RENDERER
        save_video_path = "simulate_video" if save_video_path is None else save_video_path
        makepath(save_video_path)
        sample_idx = 0 if sample_idx is None else sample_idx
        save_video_path = os.path.join(save_video_path, "{:08d}.gif".format(sample_idx))

    for step_idx in range(num_iterations):
        p.stepSimulation(physicsClientId=conn_id)
        if save_video:
            img = take_picture(renderer, conn_id=conn_id)
            images.append(img)
        if save_simul_folder:
            hand_step_path = os.path.join(save_simul_folder, "{:08d}_hand.obj".format(step_idx))
            shutil.copy(hand_tmp_fname, hand_step_path)
            obj_step_path = os.path.join(save_simul_folder, "{:08d}_obj.obj".format(step_idx))
            pos, orn = p.getBasePositionAndOrientation(obj_body_id, physicsClientId=conn_id)
            mat = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
            obj_verts_t = pos + np.dot(mat, obj_verts.T).T
            save_obj(obj_step_path, obj_verts_t, obj_faces)
        time.sleep(wait_time)
    
    if save_video:
        write_video(images, save_video_path)
        print("Saved gif to {}".format(save_video_path))
    pose_end = p.getBasePositionAndOrientation(obj_body_id, physicsClientId=conn_id)[0]

    if use_vhacd:
        os.remove(obj_tmp_fname)
    if save_obj_path is not None:
        os.remove(final_obj_tmp_fname)
    os.remove(hand_tmp_fname)
    distance = np.linalg.norm(pose_end - obj_center_mass)
    p.disconnect(physicsClientId=conn_id)

    return distance

def vhacd(
    filename,
    vhacd_path,
    resolution=1000,
    concavity=0.001,
    planeDownsampling=4,
    convexhullDownsampling=4,
    alpha=0.03,
    beta=0.0,
    maxhulls=1024,
    pca=0,
    mode=0,
    maxNumVerticesPerCH=64,
    minVolumePerCH=0.0001):

    cmd_line = (
        '"{}" --input "{}" --resolution {} --concavity {:g} '
        "--planeDownsampling {} --convexhullDownsampling {} "
        "--alpha {:g} --beta {:g} --maxhulls {:g} --pca {:b} "
        "--mode {:b} --maxNumVerticesPerCH {} --minVolumePerCH {:g} "
        '--output "{}" --log "/dev/null"'.format(
            vhacd_path,
            filename,
            resolution,
            concavity,
            planeDownsampling,
            convexhullDownsampling,
            alpha,
            beta,
            maxhulls,
            pca,
            mode,
            maxNumVerticesPerCH,
            minVolumePerCH,
            filename,
        )
    )
    print(cmd_line)
    devnull = open(os.devnull, "wb")
    vhacd_process = Popen(
        cmd_line,
        bufsize=-1,
        close_fds=True,
        shell=True,
        stdout=devnull,
        stderr=devnull
    )

    return 0 == vhacd_process.wait()


def save_obj(filename, vertices, faces):
    with open(filename, "w") as fp:
        for v in vertices:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2])) # .obj files中的顶点储存格式
        for f in faces + 1:  # Faces are 1-based, not 0-based in .obj files!
            fp.write("f %d %d %d\n" % (f[0], f[1], f[2])) # .obj files中的面储存格式


if __name__ == "__main__":
    from dataset.Dataset import GrabNetDataset
    import config

    valset = GrabNetDataset(config.dataset_dir, 'val')