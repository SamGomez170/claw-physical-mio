from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    xcarve_node = Node(
        package = "keyboard_claw_controller",
        executable = 'xcarve_controller'
    )

    ros2mqtt_bridge_node = Node(
        package = "keyboard_claw_controller",
        executable = 'ros2_mqtt_bridge'
    )




    ld.add_action(xcarve_node)
    ld.add_action(ros2mqtt_bridge_node)


    return ld