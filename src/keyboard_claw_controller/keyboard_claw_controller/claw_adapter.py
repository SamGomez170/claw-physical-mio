# Headless replacement for basic_game: opens the movement gate and forwards
# joystick commands, plus handles xcarve/goto for absolute carry/home moves.
# Reuses RosClawCtl (claw_lib) for the filter + grab path. No game rules, no UI.
#   ros2 run keyboard_claw_controller claw_adapter

import threading
import time

import paho.mqtt.client as mqtt
import rclpy

from keyboard_claw_controller.claw_lib import ClawCtl


def main(args=None):
    ctl = ClawCtl(args)
    node = ctl.ctl

    # Two enable gates, both needed or jogs get dropped silently:
    node.axis_enabled = True          # local filter flag in joystick_callback
    ctl.enable_joystick()             # joystick/enable=1, watched by xcarve_controller
    # A single enable publish races the controller's subscription (QoS depth-1).
    # Controller keeps the state once it's set, so just retry a few times at
    # startup instead of spamming forever.
    tries = [0]

    def _enable_startup():
        ctl.enable_joystick()
        tries[0] += 1
        if tries[0] >= 5:
            enable_timer.cancel()

    enable_timer = node.create_timer(1.0, _enable_startup)

    # xcarve/goto: paho thread just stashes the latest target; the drive loop
    # below runs the blocking move_to on the main thread (move_to spin_once's
    # internally, so it can't be called from the paho callback).
    pending = [None]
    lock = threading.Lock()

    def _on_connect(client, userdata, flags, reason_code, properties):
        node.get_logger().info("adapter MQTT connected rc=%s" % reason_code)
        client.subscribe("xcarve/goto", qos=0)

    def _on_message(client, userdata, msg):
        text = msg.payload.decode(errors="replace")
        parts = text.split()
        if len(parts) != 2:
            node.get_logger().warn("bad goto payload: %r" % text)
            return
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            node.get_logger().warn("bad goto payload: %r" % text)
            return
        with lock:
            pending[0] = (x, y)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect("localhost", 1883, 60)
    client.loop_start()

    node.get_logger().info("claw_adapter ready")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            with lock:
                target = pending[0]
                pending[0] = None
            if target is not None:
                node.set_home_target(*target)   # wait against this target, not (0,150)
                ctl.move_to(*target)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
        del ctl


if __name__ == '__main__':
    main()
