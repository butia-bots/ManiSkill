import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array


# TODO (stao) (xuanlin): model it properly based on real2sim
@register_agent(asset_download_ids=["widowx250s"])
class WidowX250S(BaseAgent):
    uid = "widowx250s"
    urdf_path = f"{ASSET_DIR}/robots/widowx/wx250s.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            "left_finger_link": dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            "right_finger_link": dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    ee_link_name = "ee_gripper_link"
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="top",
                pose=Pose.create_from_pq([-0.5, 0, 0.5], [1, 0, 0, 0]),
                width=256,
                height=256,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="base_link",
            ),
            CameraConfig(
                uid="wrist",
                pose=Pose.create_from_pq([-0.1, 0, 0.1], [1, 0, 0, 0]),
                width=256,
                height=256,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="ee_gripper_link",
            ),
        ]

    def _after_loading_articulation(self):
        self.finger1_link = self.robot.links_map["left_finger_link"]
        self.finger2_link = self.robot.links_map["right_finger_link"]

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    @property
    def _controller_configs(self):
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_target=True
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.1,  # a trick to have force when the object is thin
            0.1,
            self.gripper_stiffness,
            self.gripper_damping,
            normalize_action=True
        )

        controller_configs = dict(
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
        body_qvel = self.robot.get_qvel()[..., 3:-2]
        base_qvel = self.robot.get_qvel()[..., :3]
        return torch.all(body_qvel <= threshold, dim=1) & torch.all(
            base_qvel <= base_threshold, dim=1
        )

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    @property
    def tcp_pose(self) -> Pose:
        p = (self.finger1_link.pose.p + self.finger2_link.pose.p) / 2
        q = (self.finger1_link.pose.q + self.finger2_link.pose.q) / 2
        return Pose.create_from_pq(p=p, q=q)

    def is_grasping(self, object: Actor, min_force=0.0, max_angle=180):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = -self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def _after_init(self):
        self.finger1_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_finger_link"
        )
        self.finger2_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_finger_link"
        )
        self.tcp: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link"
        )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()