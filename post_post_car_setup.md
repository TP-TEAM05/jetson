# Steps to take post boot to make car operational
<!-- readd the lidar ip, if different, use the different ip address -->
sudo ip addr add 192.168.0.15/24 dev enP8p1s0
<!-- cd to new_ros -->
cd /media/anjielik2/60680b96-9018-459f-885b-5fb6f14174f1/ros2_ws/new_ros
<!-- If on another device, the appropriate path for new_ros -->
<!-- colcon build -->
colcon build
<!-- source ros -->
source /opt/ros/humble/setup.bash
<!-- source local files -->
source install/setup.bash
<!-- start the program and go -->
ros2 launch f1tenth_stack bringup_launch.py


Good luck!
