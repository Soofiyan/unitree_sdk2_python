import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/soofiyan/workspaces/unitree/unitree_sdk2_python/arm_control_ros2_ws/install/stand_arm_control'
