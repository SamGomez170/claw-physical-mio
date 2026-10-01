"""claw_adapter — headless replacement for basic_game in the command path.

basic_game is two layers: a game state machine (condition choice, scoring, pygame
UI) and the machine plumbing. The agent recreates the game layer itself, so for the
command path all we need from basic_game is the one thing it does there: open the
movement gate and let the existing filter forward joystick commands.

The filter already lives in RosClawCtl (claw_lib): it subscribes to joystick/cmd,
and when axis_enabled is True it republishes movement keys to joystick/filtered_cmd,
which xcarve_controller acts on. The only reason MQTT jogs did nothing without
basic_game is that axis_enabled inits False and nothing but basic_game ever set it
True. So this adapter reuses RosClawCtl verbatim, opens BOTH enable gates, and spins.
No game rules, no UI.

Bring-up (replaces `ros2 run ... basic_game` in terminal 2):
    ros2 run keyboard_claw_controller claw_adapter

Two gates, both must be open or jogs are silently dropped:
  1. axis_enabled      — the local filter flag in joystick_callback (set here directly;
                         enable_joystick() leaves it commented out).
  2. joystick/enable=1 — the ROS->Xcarve path xcarve_controller watches.
"""

import rclpy

from keyboard_claw_controller.claw_lib import ClawCtl


def main(args=None):
    # ClawCtl.__init__ calls rclpy.init and builds the RosClawCtl filter node.
    ctl = ClawCtl(args)
    node = ctl.ctl

    # Gate 1: the local filter flag. enable_joystick() publishes joystick/enable
    # but leaves this line commented, so set it directly or every key hits the
    # `if not self.axis_enabled: return` drop in joystick_callback.
    node.axis_enabled = True

    # Gate 2: the ROS->Xcarve enable path. Logs "Joystick movement is now ENABLED".
    ctl.enable_joystick()

    node.get_logger().info(
        "claw_adapter ready: axis_enabled=True, joystick enabled. "
        "Forwarding joystick/cmd -> joystick/filtered_cmd. No game logic."
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # ClawCtl.__del__ destroys the node and calls rclpy.shutdown().
        del ctl


if __name__ == '__main__':
    main()
