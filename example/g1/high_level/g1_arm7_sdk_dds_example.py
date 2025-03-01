import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)

import numpy as np

kPi = 3.141592654
kPi_2 = 1.57079632


from vive_tracker.track import ViveTrackerModule
from IPython import embed
from vive_tracker.fairmotion_vis import camera
from vive_tracker.fairmotion_ops import conversions, math as fairmotion_math
from vive_tracker.origin_init import (
    euler_to_matrix,
    matrix_to_euler,
    create_transformation,
    decompose_transformation,
    transform_pose,
)
import hid

VID = 0x06C2  # Replace with your VID
PID = 0x0036  # Replace with your PID
from robot_control.robot_arm_ik import G1_29_ArmIK
import pinocchio as pin


from scipy.spatial.transform import Rotation


class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20  # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21  # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28  # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29  # NOTE: Weight


class Custom:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02
        self.duration_ = 3.0
        self.counter_ = 0
        self.weight = 0.0
        self.weight_rate = 0.2
        self.kp = 60.0
        self.kd = 1.5
        self.dq = 0.0
        self.tau_ff = 0.0
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update_low_state = False
        self.crc = CRC()
        self.done = False

        self.target_pos = [
            0.0,
            kPi_2,
            0.0,
            kPi_2,
            0.0,
            0.0,
            0.0,
            0.0,
            -kPi_2,
            0.0,
            kPi_2,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
        ]

        self.target_pos = [
            -0.81947,
            0.71708,
            0.21248,
            0.76249,
            -0.14717,
            0.6742,
            -0.49733,
            -0.88907,
            -0.78951,
            -0.33882,
            1.19367,
            0.24047,
            -0.01863,
            1.02675,
            0,
            0,
            0,
        ]

        self.arm_joints = [
            G1JointIndex.LeftShoulderPitch,
            G1JointIndex.LeftShoulderRoll,
            G1JointIndex.LeftShoulderYaw,
            G1JointIndex.LeftElbow,
            G1JointIndex.LeftWristRoll,
            G1JointIndex.LeftWristPitch,
            G1JointIndex.LeftWristYaw,
            G1JointIndex.RightShoulderPitch,
            G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw,
            G1JointIndex.RightElbow,
            G1JointIndex.RightWristRoll,
            G1JointIndex.RightWristPitch,
            G1JointIndex.RightWristYaw,
            G1JointIndex.WaistYaw,
            G1JointIndex.WaistRoll,
            G1JointIndex.WaistPitch,
        ]

    def Init(self):
        # create publisher #
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        # create subscriber #
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.init_pos = None

        self.arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=False)

        self.L_tf_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0.25, +0.2, 0.1]),
        )

        self.R_tf_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0.25, -0.2, 0.1]),
        )

        self.device = hid.device()
        self.device.open(VID, PID)

        self.init_exception_flag = False

        self.init_pose_tracker_1 = None
        self.init_pose_tracker_2 = None
        self._delta_pos_l = np.zeros(3)
        self._delta_rot_l = np.zeros(3)
        self._delta_pos_r = np.zeros(3)
        self._delta_rot_r = np.zeros(3)
        self.prev_delta_pos_l = None
        self.prev_delta_rot_l = None
        self.prev_delta_pos_r = None
        self.prev_delta_rot_r = None
        self.init_delta_flag = True
        self.track_clutch_button = False
        self.track_camera_button = False

        self.zero_pose_l = np.array([0.25, 0.2, 0.2])
        self.zero_pose_r = np.array([0.25, -0.2, 0.2])

        self.l_eef_translation = np.zeros(3)
        self.r_eef_translation = np.zeros(3)

        self.rotation_speed = 0.005  # Rotation speed in radians per iteration
        self.q_target = np.zeros(35)
        self.tauff_target = np.zeros(35)

        self.scaling_factor = 0.35
        self.threshold = 0.15

        self.v_tracker = ViveTrackerModule()
        self.v_tracker.print_discovered_objects()
        self.tracker_1 = self.v_tracker.devices["tracker_1"]
        self.tracker_2 = self.v_tracker.devices["tracker_2"]

        self.camera_button_l_eef_translation = np.zeros(3)
        self.camera_button_r_eef_translation = np.zeros(3)
        self.error_camera_buton_translation = np.zeros(3)

        self.prev_l_eef_translation = np.zeros(3)
        self.prev_r_eef_translation = np.zeros(3)

        self.Kp = 1.5

        self.while_loop_init = False

        self.dt = 0.01 + self.control_dt_

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.first_update_low_state == False:
            time.sleep(1)

        if self.first_update_low_state == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.first_update_low_state == False:
            self.first_update_low_state = True

    def LowCmdWrite(self):
        self.time_ += self.control_dt_

        self._gradual_start_time = time.time()

        if self.time_ < self.duration_ * 3:
            # [Stage 1]: set robot to zero posture
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = (
                1  # 1:Enable arm_sdk, 0:Disable arm_sdk
            )
            for i, joint in enumerate(self.arm_joints):
                ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
                self.low_cmd.motor_cmd[joint].tau = 0.0
                self.low_cmd.motor_cmd[joint].q = (
                    1.0 - ratio
                ) * self.low_state.motor_state[joint].q
                self.low_cmd.motor_cmd[joint].dq = 0.0
                self.low_cmd.motor_cmd[joint].kp = self.kp
                self.low_cmd.motor_cmd[joint].kd = self.kd
        # elif self.time_ < self.duration_ * 3:
        #     # [Stage 2]: lift arms up

        #     cliped_arm_q_target = self.clip_arm_q_target(
        #         self.target_pos, velocity_limit=5.0
        #     )

        #     for i, joint in enumerate(self.arm_joints):
        #         ratio = np.clip(
        #             (self.time_ - self.duration_) / (self.duration_ * 2),
        #             0.0,
        #             1.0,
        #         )

        #         self.low_cmd.motor_cmd[joint].tau = 0.0
        #         self.low_cmd.motor_cmd[joint].q = cliped_arm_q_target[i]
        #         self.low_cmd.motor_cmd[joint].dq = 0.0
        #         self.low_cmd.motor_cmd[joint].kp = self.kp
        #         self.low_cmd.motor_cmd[joint].kd = self.kd
        elif self.time_ >= self.duration_ * 3:
            try:
                if self.init_exception_flag == False:
                    while True:
                        start_time = time.time()
                        try:
                            # Read device input
                            clutch_pressed, camera_pressed = False, False
                            input_data = 255 - self.device.read(64)[1]
                            clutch_pressed = input_data in {1, 3}
                            camera_pressed = input_data in {2, 3}

                            # Get pose data for the tracker device and format as a string
                            tracker_1_pose = np.array(
                                [val for val in self.tracker_1.get_pose_euler()]
                            )
                            tracker_2_pose = np.array(
                                [val for val in self.tracker_2.get_pose_euler()]
                            )

                            if self.init_pose_tracker_1 is None:
                                self.init_pose_tracker_1 = tracker_1_pose
                                self.init_pose_tracker_1[3:] = np.array([0, 0, 90])
                            if self.init_pose_tracker_2 is None:
                                self.init_pose_tracker_2 = tracker_2_pose
                                self.init_pose_tracker_2[3:] = np.array([0, 0, 90])

                            T_ref_inv_1 = np.linalg.inv(
                                create_transformation(self.init_pose_tracker_1)
                            )
                            T_ref_inv_2 = np.linalg.inv(
                                create_transformation(self.init_pose_tracker_2)
                            )

                            tracker_1_pose = np.array(
                                transform_pose(tracker_1_pose, T_ref_inv_1)
                            )
                            tracker_2_pose = np.array(
                                transform_pose(tracker_2_pose, T_ref_inv_2)
                            )

                            tracker_1_pose[3:] = [
                                tracker_1_pose[5],
                                -tracker_1_pose[3],
                                -tracker_1_pose[4],
                            ]
                            tracker_1_pose[:3] = [
                                -tracker_1_pose[1],
                                -tracker_1_pose[2],
                                tracker_1_pose[0],
                            ]

                            tracker_2_pose[3:] = [
                                -tracker_2_pose[5],
                                -tracker_2_pose[3],
                                tracker_2_pose[4],
                            ]
                            tracker_2_pose[:3] = [
                                -tracker_2_pose[1],
                                -tracker_2_pose[2],
                                tracker_2_pose[0],
                            ]

                            self.curr_delta_pos_r, self.curr_delta_rot_r = (
                                tracker_2_pose[:3],
                                tracker_2_pose[3:],
                            )
                            self.curr_delta_pos_l, self.curr_delta_rot_l = (
                                tracker_1_pose[:3],
                                tracker_1_pose[3:],
                            )

                            if not clutch_pressed:
                                self.curr_delta_pos_r, self.curr_delta_rot_r = (
                                    tracker_2_pose[:3],
                                    tracker_2_pose[3:],
                                )
                                self.curr_delta_pos_l, self.curr_delta_rot_l = (
                                    tracker_1_pose[:3],
                                    tracker_1_pose[3:],
                                )

                            if self.track_clutch_button and not clutch_pressed:
                                self.prev_delta_pos_r, self.prev_delta_rot_r = (
                                    self.curr_delta_pos_r.copy(),
                                    self.curr_delta_rot_r.copy(),
                                )
                                self.prev_delta_pos_l, self.prev_delta_rot_l = (
                                    self.curr_delta_pos_l.copy(),
                                    self.curr_delta_rot_l.copy(),
                                )
                                print("clutch accessed")
                                self.track_clutch_button = False

                            if self.init_delta_flag:
                                self.prev_delta_pos_l, self.prev_delta_rot_l = (
                                    self.curr_delta_pos_l.copy(),
                                    self.curr_delta_rot_l.copy(),
                                )
                                self.prev_delta_pos_r, self.prev_delta_rot_r = (
                                    self.curr_delta_pos_r.copy(),
                                    self.curr_delta_rot_r.copy(),
                                )
                                self.init_delta_flag = False

                            if clutch_pressed:
                                self.curr_delta_pos_r, self.curr_delta_rot_r = (
                                    self.prev_delta_pos_r.copy(),
                                    self.prev_delta_rot_r.copy(),
                                )
                                self.curr_delta_pos_l, self.curr_delta_rot_l = (
                                    self.prev_delta_pos_l.copy(),
                                    self.prev_delta_rot_l.copy(),
                                )
                                self.track_clutch_button = True

                            if camera_pressed and not self.track_camera_button:
                                self.camera_button_l_eef_translation = (
                                    self.l_eef_translation
                                )
                                self.camera_button_r_eef_translation = (
                                    self.r_eef_translation
                                )
                                self.error_camera_buton_translation = (
                                    self.camera_button_l_eef_translation
                                    - self.camera_button_r_eef_translation
                                )
                                self.track_camera_button = True
                            elif not camera_pressed and self.track_camera_button:
                                self.track_camera_button = False

                            if camera_pressed:
                                self._delta_pos_l += (
                                    self.curr_delta_pos_l - self.prev_delta_pos_l
                                )
                                self._delta_rot_l += (
                                    self.curr_delta_rot_l - self.prev_delta_rot_l
                                )
                                self._delta_pos_r += (
                                    self.curr_delta_pos_l - self.prev_delta_pos_l
                                )
                                self._delta_rot_r += (
                                    self.curr_delta_rot_l - self.prev_delta_rot_l
                                )
                            else:
                                self._delta_pos_l += (
                                    self.curr_delta_pos_l - self.prev_delta_pos_l
                                )
                                self._delta_rot_l += (
                                    self.curr_delta_rot_l - self.prev_delta_rot_l
                                )
                                self._delta_pos_r += (
                                    self.curr_delta_pos_r - self.prev_delta_pos_r
                                )
                                self._delta_rot_r += (
                                    self.curr_delta_rot_r - self.prev_delta_rot_r
                                )

                            self.prev_delta_pos_l, self.prev_delta_rot_l = (
                                self.curr_delta_pos_l.copy(),
                                self.curr_delta_rot_l.copy(),
                            )
                            self.prev_delta_pos_r, self.prev_delta_rot_r = (
                                self.curr_delta_pos_r.copy(),
                                self.curr_delta_rot_r.copy(),
                            )

                            self.L_tf_target.translation = (
                                self.zero_pose_l + self._delta_pos_l
                            )
                            self.L_tf_target.rotation = Rotation.from_euler(
                                "xyz", self._delta_rot_l, degrees=True
                            ).as_matrix()

                            self.R_tf_target.translation = (
                                self.zero_pose_r + self._delta_pos_r
                            )

                            self.R_tf_target.rotation = Rotation.from_euler(
                                "xyz", self._delta_rot_r, degrees=True
                            ).as_matrix()

                            l_error = (
                                self.L_tf_target.translation - self.l_eef_translation
                            )
                            r_error = (
                                self.R_tf_target.translation - self.r_eef_translation
                            )

                            # self.L_tf_target.translation += self.Kp * l_error
                            # self.R_tf_target.translation += self.Kp * r_error

                            l_velocity_error = (
                                self.l_eef_translation - self.prev_l_eef_translation
                            ) / self.dt
                            r_velocity_error = (
                                self.r_eef_translation - self.prev_r_eef_translation
                            ) / self.dt

                            # if camera_pressed:
                            #     self.R_tf_target.translation += (
                            #         self.Kp * r_error
                            #         + 5
                            #         * (
                            #             (self.l_eef_translation - self.r_eef_translation)
                            #             - self.error_camera_buton_translation
                            #         )
                            #         + 0.1 * (l_velocity_error - r_velocity_error)
                            #     )

                        except Exception as e:
                            print("error in controller", e)

                        current_lr_arm_q = self.get_current_dual_arm_q()[:-3]
                        current_lr_arm_dq = self.get_current_dual_arm_dq()[:-3]
                        current_lr_arm_tau_est = self.get_current_dual_arm_tau_est()[
                            :-3
                        ]

                        current_lr_arm_q = np.delete(current_lr_arm_q, 5)
                        current_lr_arm_dq = np.delete(current_lr_arm_dq, 5)
                        current_lr_arm_tau_est = np.delete(current_lr_arm_tau_est, 5)

                        sol_q, sol_tauff, l_eef, r_eef = self.arm_ik.solve_ik(
                            self.L_tf_target.homogeneous,
                            self.R_tf_target.homogeneous,
                            current_lr_arm_q,
                            current_lr_arm_dq,
                            current_lr_arm_tau_est,
                        )
                        self.prev_l_eef_translation = self.l_eef_translation
                        self.prev_r_eef_translation = self.r_eef_translation

                        self.l_eef_translation = l_eef.translation
                        self.r_eef_translation = r_eef.translation

                        sol_q = np.insert(sol_q, 5, 0)
                        sol_tauff = np.insert(sol_tauff, 5, 0)

                        self.target_pos = sol_q
                        self.target_pos = np.append(self.target_pos, [0, 0, 0])
                        sol_tauff = np.append(sol_tauff, [0, 0, 0])

                        # [Stage 3]: set robot back to zero posture
                        cliped_arm_q_target = self.clip_arm_q_target(
                            self.target_pos, velocity_limit=5.0
                        )

                        for i, joint in enumerate(self.arm_joints):
                            ratio = np.clip(
                                (self.time_ - self.duration_) / (self.duration_ * 2),
                                0.0,
                                1.0,
                            )

                            self.low_cmd.motor_cmd[joint].tau = sol_tauff[i]
                            self.low_cmd.motor_cmd[joint].q = cliped_arm_q_target[i]
                            self.low_cmd.motor_cmd[joint].dq = 0.0
                            self.low_cmd.motor_cmd[joint].kp = self.kp
                            self.low_cmd.motor_cmd[joint].kd = self.kd

                        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                        self.arm_sdk_publisher.Write(self.low_cmd)

                        t_elapsed = start_time - self._gradual_start_time
                        self.arm_velocity_limit = 5.0 + (
                            10.0 * min(1.0, t_elapsed / 5.0)
                        )

                        current_time = time.time()
                        all_t_elapsed = current_time - start_time
                        sleep_time = max(0, (self.control_dt_ - all_t_elapsed))
                        time.sleep(sleep_time)

            except KeyboardInterrupt:
                if self.init_exception_flag == False:
                    self.time_ = 0.0
                    self.init_exception_flag = True

        if self.time_ < self.duration_ * 3:
            # [Stage 3]: set robot back to zero posture
            for i, joint in enumerate(self.arm_joints):
                ratio = np.clip(
                    (self.time_) / (self.duration_ * 3),
                    0.0,
                    1.0,
                )
                self.low_cmd.motor_cmd[joint].tau = 0.0
                self.low_cmd.motor_cmd[joint].q = (
                    1.0 - ratio
                ) * self.low_state.motor_state[joint].q
                self.low_cmd.motor_cmd[joint].dq = 0.0
                self.low_cmd.motor_cmd[joint].kp = self.kp
                self.low_cmd.motor_cmd[joint].kd = self.kd

        elif self.time_ < self.duration_ * 4:
            # [Stage 4]: release arm_sdk
            for i, joint in enumerate(self.arm_joints):
                ratio = np.clip(
                    (self.time_ - self.duration_ * 3) / (self.duration_),
                    0.0,
                    1.0,
                )
                self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = (
                    1 - ratio
                )  # 1:Enable arm_sdk, 0:Disable arm_sdk
        if self.init_exception_flag == True:
            if self.time_ < self.duration_:
                # [Stage 4]: release arm_sdk
                for i, joint in enumerate(self.arm_joints):
                    ratio = np.clip(
                        (self.time_) / (self.duration_),
                        0.0,
                        1.0,
                    )
                    self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1 - ratio
            else:
                self.done = True

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

    def get_current_dual_arm_q(self):
        return np.array([self.low_state.motor_state[id].q for id in self.arm_joints])

    def get_current_dual_arm_dq(self):
        return np.array([self.low_state.motor_state[id].dq for id in self.arm_joints])

    def get_current_dual_arm_tau_est(self):
        return np.array(
            [self.low_state.motor_state[id].tau_est for id in self.arm_joints]
        )

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt_)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target


if __name__ == "__main__":

    print(
        "WARNING: Please ensure there are no obstacles around the robot while running this example."
    )
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    while True:
        time.sleep(1)
        if custom.done:
            print("Done!")
            sys.exit(-1)
