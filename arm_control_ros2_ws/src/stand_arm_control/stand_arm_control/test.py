

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

import numpy as np
from scipy.spatial.transform import Rotation as R

class TFListener(Node):
    def __init__(self):
        super().__init__('tf_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.get_transform)  # 10 Hz update rate
        self.vive_trans = None
        self.vive_rot = None

    def get_transform(self):
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                'libsurvive_world',  # Parent frame
                'LHR-9180C0A1',      # Child frame
                rclpy.time.Time()
            )
            translation = trans.transform.translation
            quaternion = trans.transform.rotation
            self.vive_trans = np.array([translation.x, translation.y, translation.z])
            self.vive_rot = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
            self.vive_rot = R.from_quat(self.vive_rot).as_euler('xyz', degrees=True)
            # self.get_logger().info(f'Translation: {trans.transform.translation}')
            # self.get_logger().info(f'Rotation: {trans.transform.rotation}')
        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {str(e)}')

rclpy.init()
node = TFListener()

while rclpy.ok():
    rclpy.spin_once(node)
    print(node.vive_trans, node.vive_rot)

