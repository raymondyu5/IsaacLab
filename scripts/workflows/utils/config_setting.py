from isaaclab.assets import RigidObjectCfg, DeformableObjectCfg, AssetBaseCfg, ArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, DeformableBodyPropertiesCfg
from isaaclab.sim.spawners import UsdFileCfg
import isaaclab.utils.math as math_utils
import torch

from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg
from isaaclab.utils.math import quat_from_matrix, create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
import json
import numpy as np

from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
import os


def get_transform_from_txt(txt_path: str):
    with open(txt_path, "r") as f:
        lines = f.read()
    lines = lines.split("\n")
    mat = [l.split(" ") for l in lines]
    mat = mat[:-1]
    mat = np.array(mat).astype(np.float32)
    return mat


def get_scale_from_transform(t: np.ndarray):
    return np.linalg.det(t[:3, :3])**(1 / 3)


def get_parameters_from_intrinsics_matrix(
    camera_matrix,
    width,
    height,
    pixel_size=3 * 1e-3,
):
    ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix
    horizontal_aperture = pixel_size * width  # The aperture size in mm
    vertical_aperture = pixel_size * height
    focal_length_x = fx * pixel_size
    focal_length_y = fy * pixel_size
    focal_length = (focal_length_x +
                    focal_length_y) / 2  # The focal length in mm
    return focal_length, horizontal_aperture, vertical_aperture


def scale_intrinsics(K_orig, new_width, new_height, orig_width, orig_height):
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height

    fx_new = K_orig[0][0] * scale_x
    fy_new = K_orig[1][1] * scale_y
    cx_new = K_orig[0][2] * scale_x
    cy_new = K_orig[1][2] * scale_y

    K_new = [[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]]
    return np.array(K_new)


def load_camera_params(camera_name, camera_setting, camera_json):

    if camera_json is not None:

        with open(camera_json, 'r') as file:
            camera_data = json.load(file)
        from scipy.spatial.transform import Rotation as R

        for cam_name in camera_data.keys():
            if cam_name == camera_name:
                camera_cfg = camera_data[cam_name]
                break

        camera_translate = torch.tensor(camera_cfg["extrinsic"]["position"])
        cam_quat = torch.as_tensor(
            R.from_euler(
                "xyz",
                camera_cfg["extrinsic"]["rotation_rpy"],
            ).as_quat()[[3, 0, 1, 2]]).to(torch.float32)
        camera_orient = math_utils.matrix_from_quat(cam_quat)

        camera_setting["cam_quat"] = cam_quat.cpu().numpy().tolist()
        camera_setting["cam_trans"] = camera_translate.cpu().numpy().tolist()

        transformation_matrix = torch.tensor([[1.0, 0.0,
                                               0.0], [0.0, -1.0, 0.0],
                                              [0.0, 0.0,
                                               -1.0]]).to(torch.float32)

        # Perform matrix multiplication
        camera_orient = camera_orient @ transformation_matrix
        camera_rot = quat_from_matrix(camera_orient)

        K = np.array(camera_cfg["intrinsic"]["camera_matrix"])

        K_scale = scale_intrinsics(K, camera_setting["width"],
                                   camera_setting["height"],
                                   camera_cfg["width"], camera_cfg["height"])

        camera_setting["width"] = camera_setting["width"]
        camera_setting["height"] = camera_setting["height"]

        focal_length, horizontal_aperture, vertical_aperture = get_parameters_from_intrinsics_matrix(
            camera_matrix=K_scale,
            width=camera_setting["width"],
            height=camera_setting["height"],
        )

        camera_spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=K_scale.reshape(-1),
            width=camera_setting["width"],
            height=camera_setting["height"],
            focal_length=focal_length * 10,
            focus_distance=400.0,
            clipping_range=(0.1, 1.0e5),
        )
        camera_setting["focal_length"] = focal_length * 10

    else:

        if "eyes" in camera_setting.keys():
            camera_rot = quat_from_matrix(
                create_rotation_matrix_from_view(
                    torch.as_tensor([camera_setting["eyes"]]),
                    torch.as_tensor([camera_setting["target"]])))[0]
            camera_translate = camera_setting["eyes"]
        else:
            if "quat" in camera_setting.keys():
                camera_rot = torch.as_tensor(camera_setting["quat"])
            elif "rotation" in camera_setting.keys():
                camera_rot = torch.as_tensor(camera_setting["rotation"])
                camera_rot = math_utils.quat_from_matrix(camera_rot)

            else:
                euler = torch.as_tensor([camera_setting["euler"]
                                         ]) / 180.0 * torch.pi

                camera_rot = math_utils.quat_from_euler_xyz(
                    euler[:, 0], euler[:, 1], euler[:, 2])[0]

            camera_translate = camera_setting["translate"]
            # if camera_setting.get("converter") is not None:
            #     camera_rot = math_utils.convert_camera_frame_orientation_convention(
            #         camera_rot,
            #         origin=camera_setting["converter"],
            #         target="opengl")

        camera_spawn = sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            focus_distance=400.0,
            horizontal_aperture=20,
            clipping_range=(0.1, 1.0e5),
        )
    return camera_rot, camera_translate, camera_spawn, camera_setting


def configure_deformable_object(scene, object_name, object_config,
                                current_path):

    deformable_cfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + object_name,
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=object_config['pos'],
            rot=tuple(
                obtain_target_quat_from_multi_angles(
                    object_config["rot"]["axis"],
                    object_config["rot"]["angles"]))),
        spawn=UsdFileCfg(
            usd_path=f"{current_path}/{object_config['path']}",
            scale=object_config['scale'],
            mass_props=sim_utils.MassPropertiesCfg(mass=200),
            deformable_props=DeformableBodyPropertiesCfg(
                simulation_hexahedral_resolution=object_config.get(
                    "simulation_hexahedral_resolution"),
                vertex_velocity_damping=object_config.get(
                    "vertex_velocity_damping"),
            ),
            semantic_tags=[("class", "deformable")],
        ),
        deform_cfg=object_config,
    )

    setattr(scene, object_name, deformable_cfg)


def load_rigid_object_from_folder(scene, object_config, current_path, pos, rot,
                                  scale):

    objects_name = os.listdir(object_config["load_from_folder"])
    for index, object_name in enumerate(objects_name):
        usd_path = os.path.join(
            f"{current_path}/" + object_config["load_from_folder"],
            object_name + f"/{object_config['path']}")
        pos[1] += index * 0.15

        rigid_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + object_name,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=UsdFileCfg(usd_path=usd_path,
                             scale=scale,
                             mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                             rigid_props=RigidBodyPropertiesCfg(
                                 solver_position_iteration_count=16,
                                 solver_velocity_iteration_count=1,
                                 max_angular_velocity=1000.0,
                                 max_linear_velocity=1000.0,
                                 disable_gravity=False,
                                 kinematic_enabled=object_config.get(
                                     "kinematic_enabled", False),
                             ),
                             semantic_tags=[("class", object_name)]),
            rigid_cfg=object_config,
        )
        setattr(scene, object_name, rigid_cfg)


def configure_multi_articulation_object(
    scene,
    articulation_objects,
    current_path,
    spawn_multi_assets_name,
):

    usd_paths = []

    for object_name, object_config in articulation_objects.items():

        usd_path = f"{current_path}/{object_config['path']}"
        usd_paths.append(usd_path)
    rot = tuple(
        obtain_target_quat_from_multi_angles(object_config["rot"]["axis"],
                                             object_config["rot"]["angles"]))

    multi_articulation_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}" + f"/{spawn_multi_assets_name}",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=False,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=object_config["disable_gravity"]
                if "disable_gravity" in object_config.keys() else False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=object_config[
                    "enabled_self_collisions"] if "enabled_self_collisions"
                in object_config.keys() else False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0),
            scale=object_config["scale"],
            semantic_tags=[("class", object_name)]),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=object_config["pos"],
            rot=rot,
            joint_pos=object_config["joints"]),
        actuators={
            f"{object_name}":
            ImplicitActuatorCfg(
                joint_names_expr=[*object_config["joints"].keys()],
                effort_limit=200.0,
                velocity_limit=300.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
            ),
        },
    )

    setattr(scene, spawn_multi_assets_name, multi_articulation_cfg)

    # rigid_cfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}" + f"/{spawn_multi_assets_name}/" +
    #     articulation_link,
    #     spawn=None,
    # )
    # setattr(scene, articulation_link.split("/")[-1], rigid_cfg)


def configure_multi_cluster_rigid_object(
    scene,
    rigid_objects,
    current_path,
    add_left_hand,
    add_right_hand,
    multi_cluster_rigid,
):

    loaded_objects = []

    for cluster_name, cluster_name_config in multi_cluster_rigid.items():

        handness = cluster_name_config.get("handness", None)
        if handness is not None:
            if not eval(f"add_{handness}_hand"):
                continue

        cluster_objects = cluster_name_config["objects_list"]
        usd_paths = []
        loaded_objects += cluster_objects

        for object_name in cluster_objects:

            usd_path = f"{current_path}/{rigid_objects[object_name]['path']}"
            usd_paths.append(usd_path)
        rot = tuple(
            obtain_target_quat_from_multi_angles(
                cluster_name_config["rot"]["axis"],
                cluster_name_config["rot"]["angles"]))
        kinematic_enabled = cluster_name_config.get("kinematic_enabled", False)

        multi_rigid_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}" + f"/{cluster_name}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=cluster_name_config["pos"], rot=rot),
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=usd_paths,
                random_choice=False,
                scale=cluster_name_config["scale"],
                mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                    kinematic_enabled=kinematic_enabled),
                semantic_tags=[("class", cluster_name)],
            ),
        )
        setattr(scene, cluster_name, multi_rigid_cfg)
    return loaded_objects


def configure_multi_rigid_object(scene, rigid_objects, current_path,
                                 spawn_multi_assets_name):
    usd_paths = []
    for object_name, object_config in rigid_objects.items():

        usd_path = f"{current_path}/{object_config['path']}"
        usd_paths.append(usd_path)
    rot = tuple(
        obtain_target_quat_from_multi_angles(object_config["rot"]["axis"],
                                             object_config["rot"]["angles"]))

    multi_rigid_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}" + f"/{spawn_multi_assets_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.15], rot=rot),
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=False,
            scale=object_config["scale"],
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )
    setattr(scene, spawn_multi_assets_name, multi_rigid_cfg)


def configure_rigid_object(scene, object_name, object_config, current_path):

    if object_config.get("spawn", True):
        if object_config.get("transformation_txt") is None:
            if object_config.get("rot") is not None:
                rot = tuple(
                    obtain_target_quat_from_multi_angles(
                        object_config["rot"]["axis"],
                        object_config["rot"]["angles"]))
            elif object_config.get("quat") is not None:
                rot = object_config["quat"]
            pos = torch.as_tensor(object_config['pos'], dtype=torch.float16)

            scale = object_config["scale"]

        else:
            mat = get_transform_from_txt(object_config["transformation_txt"])
            pos = mat[:3, 3]
            rot = math_utils.quat_from_matrix(torch.as_tensor(
                mat[:3, :3])).numpy()
            scale = np.ones(3) * get_scale_from_transform(mat)
            object_config["scale"] = scale
            object_config["rot"]["quat"] = rot
            object_config["rot"]["pos"] = pos

        if object_config.get("load_from_folder") is not None:
            load_rigid_object_from_folder(scene, object_config, current_path,
                                          pos, rot, scale)
        else:

            rigid_cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + object_name,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                spawn=UsdFileCfg(
                    usd_path=f"{current_path}/{object_config['path']}",
                    scale=scale,
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        disable_gravity=False,
                        kinematic_enabled=object_config.get(
                            "kinematic_enabled", False),
                    ),
                    semantic_tags=[("class", object_name)]),
                rigid_cfg=object_config,
            )
            setattr(scene, object_name, rigid_cfg)

    else:
        scale = None

        rigid_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/" + object_config['path'],
            spawn=None,
        )
        setattr(scene, object_name, rigid_cfg)

    return object_config


def configure_articulation_object(scene, object_name, object_config,
                                  current_path):
    articulate_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}" + f"/{object_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{current_path}/{object_config['path']}",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=object_config["disable_gravity"]
                if "disable_gravity" in object_config.keys() else False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=object_config[
                    "enabled_self_collisions"] if "enabled_self_collisions"
                in object_config.keys() else False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0),
            scale=object_config["scale"],
            semantic_tags=[("class", object_name)]),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=object_config["pos"],
            rot=tuple(
                obtain_target_quat_from_multi_angles(
                    object_config["rot"]["axis"],
                    object_config["rot"]["angles"])),
            joint_pos=object_config["joints"]),
        actuators={
            f"{object_name}":
            ImplicitActuatorCfg(
                joint_names_expr=[*object_config["joints"].keys()],
                effort_limit=200.0,
                velocity_limit=300.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
            ),
        },
        articulation_cfg=object_config)

    setattr(scene, object_name, articulate_cfg)


def configure_asset_object(scene, object_name, object_config, current_path):
    asset_cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + object_name,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=object_config['pos'],
            rot=tuple(
                obtain_target_quat_from_multi_angles(
                    object_config["rot"]["axis"],
                    object_config["rot"]["angles"]))),
        spawn=UsdFileCfg(
            usd_path=f"{current_path}/{object_config['path']}",
            scale=object_config['scale'],
        ),
    )
    setattr(scene, object_name, asset_cfg)


def configure_camera(scene,
                     camera_name,
                     camera_setting,
                     customize_camera_setting,
                     randomness=None,
                     camera_json=None):
    camera_rot, camera_translate, camera_spawn, camera_setting = load_camera_params(
        camera_name, camera_setting, camera_json)

    if camera_setting.get("prim_path", None) is not None:
        prim_path = "{ENV_REGEX_NS}/" + camera_setting["prim_path"]
        offset = CameraCfg.OffsetCfg(pos=camera_translate,
                                     rot=camera_rot,
                                     convention="ros")
    else:
        prim_path = "{ENV_REGEX_NS}/" + camera_name
        offset = CameraCfg.OffsetCfg(pos=camera_translate,
                                     rot=camera_rot,
                                     convention="opengl")

    camera_cfg = CameraCfg(
        prim_path=prim_path,
        offset=offset,
        update_period=camera_setting["update_period"],
        height=camera_setting["height"],
        width=camera_setting["width"],
        data_types=camera_setting["data_types"],
        spawn=camera_spawn,
        # segmentation_point_cloud=customize_camera_setting[
        #     "segmentation_point_cloud"],
        # colorize_point_cloud=customize_camera_setting["colorize_point_cloud"],
        # num_downsampled_points=customize_camera_setting[
        #     "num_downsampled_points"],
        # segmentation_point_cloud_id=customize_camera_setting[
        #     "segmentation_point_cloud_id"],
        debug_vis=True,
        colorize_instance_segmentation=False)

    setattr(scene, camera_name, camera_cfg)
    return camera_setting


def configure_raycaster_camera(scene,
                               camera_name,
                               camera_setting,
                               randomness=None,
                               camera_json=None):
    camera_rot, camera_translate, camera_spawn, camera_setting = load_camera_params(
        camera_name, camera_setting, camera_json)

    camera_cfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/" + "right_hand",
        offset=RayCasterCfg.OffsetCfg(
            pos=camera_translate,
            rot=camera_rot,
        ),
        mesh_prim_paths=["/World/GroundPlane"],
        update_period=1 / 60,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.0)),
        attach_yaw_only=True,
    )
    setattr(scene, camera_name, camera_cfg)


def configure_tiled_camera(scene,
                           camera_name,
                           camera_setting,
                           randomness=None,
                           camera_json=None):
    camera_rot, camera_translate, camera_spawn, _ = load_camera_params(
        camera_name, camera_setting, camera_json)

    if camera_setting.get("prim_path", None) is not None:
        prim_path = "/World/Camera"
        offset = CameraCfg.OffsetCfg(pos=camera_translate,
                                     rot=camera_rot,
                                     convention="opengl")

    else:
        prim_path = "{ENV_REGEX_NS}/" + camera_name
        offset = CameraCfg.OffsetCfg(pos=camera_translate,
                                     rot=camera_rot,
                                     convention="opengl")

    camera_cfg = TiledCameraCfg(
        prim_path=prim_path,
        offset=offset,
        update_period=camera_setting["update_period"],
        height=camera_setting["height"],
        width=camera_setting["width"],
        data_types=camera_setting["data_types"],
        spawn=camera_spawn,
        customize_camera_setting=randomness if randomness is not None else {},
    )
    setattr(scene, camera_name, camera_cfg)


def configure_particle_object(scene, object_name, object_config, current_path):
    particle_cfg = ParticleObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + {object_name},
        init_state=ParticleObjectCfg.InitialStateCfg(
            pos=object_config['pos'],
            rot=tuple(
                obtain_target_quat_from_multi_angles(
                    object_config["rot"]["axis"],
                    object_config["rot"]["angles"]))),
        spawn=UsdFileCfg(
            usd_path=f"{current_path}/{object_config['path']}",
            scale=object_config['scale'],
            particle_props=ParticleBodyPropertiesCfg(),
        ),
        deform_cfg=object_config,
    )
    setattr(scene, object_name, particle_cfg)


def set_from_config(scene, env_cfg, current_path):

    deformable_objects = env_cfg['params'].get('DeformableObject', {})
    if deformable_objects is not None:
        for object_name, object_config in deformable_objects.items():
            configure_deformable_object(scene, object_name, object_config,
                                        current_path)

    # Configure Asset Objects
    asset_objects = env_cfg['params'].get('AssstObject', {})
    if asset_objects is not None:
        for object_name, object_config in asset_objects.items():
            configure_asset_object(scene, object_name, object_config,
                                   current_path)

    # Configure Articulation Objects
    articulation_objects = env_cfg['params'].get('ArticulationObject', {})

    if articulation_objects is not None:

        if not env_cfg['params'].get("spawn_multi_articulation", False):
            for object_name, object_config in articulation_objects.items():
                configure_articulation_object(scene, object_name,
                                              object_config, current_path)
        else:

            configure_multi_articulation_object(
                scene, articulation_objects, current_path,
                env_cfg['params']["spawn_multi_articulation"]["name"])

    # Configure Rigid Objects
    rigid_objects = env_cfg['params'].get('RigidObject', {})

    if rigid_objects is not None:
        loaded_objects = []

        if env_cfg['params'].get("multi_cluster_rigid", None):
            loaded_objects = configure_multi_cluster_rigid_object(
                scene,
                rigid_objects,
                current_path,
                env_cfg['params']["add_left_hand"],
                env_cfg['params']["add_right_hand"],
                env_cfg['params']["multi_cluster_rigid"],
            )

        for object_name, object_config in rigid_objects.items():
            if object_name in loaded_objects:
                continue

            configure_rigid_object(scene, object_name, object_config,
                                   current_path)

        # else:
        #     configure_multi_rigid_object(
        #         scene, rigid_objects, current_path,
        #         env_cfg['params']["spawn_multi_assets_name"])

    # Configure Cameras

    if "Camera" in env_cfg['params'].keys():
        if env_cfg['params']['Camera'].get("initial", False):
            extract_seg_pc = env_cfg['params']['Camera'].get(
                "extract_seg_pc", False)
            extract_rgb = env_cfg['params']['Camera'].get("extract_rgb", False)
            extrat_seg_rgb = env_cfg['params']['Camera'].get(
                "extract_seg_rgb", False)
            extract_all_pc = env_cfg['params']['Camera'].get(
                "extract_all_pc", False)

            for camera_name, camera_setting in env_cfg['params']['Camera'][
                    "cameras"].items():

                camera_type = camera_setting["camera_type"]

                if not extract_seg_pc and not extrat_seg_rgb:
                    if "semantic_segmentation" in camera_setting["data_types"]:
                        camera_setting["data_types"].remove(
                            "semantic_segmentation")
                if not extract_seg_pc and not extract_all_pc:

                    if "distance_to_image_plane" in camera_setting[
                            "data_types"]:
                        camera_setting["data_types"].remove(
                            "distance_to_image_plane")
                if not extract_rgb and not extrat_seg_rgb:
                    if "rgb" in camera_setting["data_types"]:

                        camera_setting["data_types"].remove("rgb")

                if camera_type == "camera":
                    camera_setting = configure_camera(
                        scene,
                        camera_name,
                        camera_setting,
                        customize_camera_setting=env_cfg['params']['Camera']
                        ["pointcloud_setting"],
                        randomness=env_cfg['params']['Camera']
                        ["random_pose_setting"]
                        if env_cfg['params']['Camera'].get(
                            "random_pose", False) else None,
                        camera_json=env_cfg['params']['Camera']["camera_json"])
                    env_cfg['params']['Camera'][camera_name] = camera_setting

                elif camera_type == "tiled_camera":
                    configure_tiled_camera(
                        scene,
                        camera_name,
                        camera_setting,
                        randomness=env_cfg['params']['Camera']
                        ["random_pose_setting"]
                        if env_cfg['params']['Camera'].get(
                            "random_pose", False) else None,
                        camera_json=env_cfg['params']['Camera']["camera_json"])
                elif camera_type == "ray_caster_camera":
                    configure_raycaster_camera(
                        scene,
                        camera_name,
                        camera_setting,
                        randomness=env_cfg['params']['Camera']
                        ["random_pose_setting"]
                        if env_cfg['params']['Camera'].get(
                            "random_pose", False) else None,
                        camera_json=env_cfg['params']['Camera']["camera_json"])

    particle_objects = env_cfg['params'].get('ParticlesObject', {})
    if particle_objects is not None:
        for object_name, object_config in particle_objects.items():
            configure_particle_object(scene, object_name, object_config,
                                      current_path)
