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


from oculus.oculus_reader import OculusReader
from robot_control.robot_arm_ik_orig import G1_29_ArmIK
import pinocchio as pin


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

        self.oculus_reader = OculusReader()
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
        self._delta_pos = None
        self.zero_pose = np.array([0.25, -0.2, 0.1])
        self.init_exception_flag = False

        self.while_loop_init = False

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

        oculus_readings = (
            self.oculus_reader.get_transformations_and_buttons()
        )
        self._gradual_start_time = time.time()

        if self.time_ < self.duration_*3:
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
        elif self.time_ >= self.duration_*3:
            try:
                if self.init_exception_flag == False:
                    while True:
                        start_time = time.time()
                        current_lr_arm_q = self.get_current_dual_arm_q()[:-3]
                        current_lr_arm_dq = self.get_current_dual_arm_dq()[:-3]
                        current_lr_arm_tau_est = self.get_current_dual_arm_tau_est()[
                            :-3
                        ]
                        print("current g", current_lr_arm_tau_est)
                        try:
                            oculus_readings = (
                                self.oculus_reader.get_transformations_and_buttons()
                            )
                            # print(oculus_readings)

                            teleop_rot = oculus_readings[0]["r"][:3, :3]
                            trans = oculus_readings[0]["r"][:3, 3]
                            trans = np.array([-trans[2], -trans[0], trans[1]])

                            index_finger = oculus_readings[1]["rightTrig"][0]
                            thumb_finger_1 = oculus_readings[1]["RThU"]
                            thumb_finger_2 = oculus_readings[1]["A"]
                            middle_finger = oculus_readings[1]["rightGrip"][0]

                            if self.init_pos is None:
                                self.init_pos = trans
                            self._delta_pos = trans - self.init_pos

                            self.R_tf_target.translation = (
                                self.zero_pose + self._delta_pos
                            )
                            
                        except Exception as e:
                            print("error in controller", e)
                            oculus_readings = None

                        sol_q, sol_tauff = self.arm_ik.solve_ik(
                            self.L_tf_target.homogeneous,
                            self.R_tf_target.homogeneous,
                            current_lr_arm_q,
                            current_lr_arm_dq,
                            current_lr_arm_tau_est,
                        )

                        self.target_pos = sol_q
                        self.target_pos = np.append(self.target_pos, [0, 0, 0])
                        sol_tauff = np.append(sol_tauff, [0, 0, 0])

                        print("target g", sol_tauff)

                        # [Stage 3]: set robot back to zero posture
                        cliped_arm_q_target = self.clip_arm_q_target(
                            self.target_pos, velocity_limit=20.0
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
                        self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

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
