import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8,String
from claw_machine_msgs.msg import Position
import threading
import time


class ClawCtl():
    '''Wrapper class for using ROS2 functions for claw controller'''
    def __init__(self, args):
        rclpy.init(args=args)
        self.ctl = RosClawCtl()

    def disable_joystick(self):
        #disable joystick
        self.ctl.get_logger().info(f'disabling joystick...')
        joytick_enable_msg = UInt8()
        joytick_enable_msg.data = 0
        self.ctl.joystick_enable_publisher.publish(joytick_enable_msg)
    
    def enable_joystick(self):
        #enable joystick to send commands
        self.ctl.get_logger().info(f'enabling joystick ...')
        joytick_enable_msg = UInt8()
        joytick_enable_msg.data = 1
        self.ctl.joystick_enable_publisher.publish(joytick_enable_msg)

    def move_home(self):
        #move xcarve to initial position
        self.ctl.get_logger().info(f'going to home position...')
        xcarve_position_msg = Position()
        xcarve_position_msg.x = 0.0
        xcarve_position_msg.y = 150.0
        self.ctl.xcarve_goto_publisher.publish(xcarve_position_msg)

        #wait to get to home position
        self.ctl.home_event.clear()
        while not self.ctl.home_event.is_set():
            rclpy.spin_once(self.ctl, timeout_sec=0.5)

        self.ctl.get_logger().info(f'home position.')

    def wait_fire_button(self):
        #wait for red button to be pressed
        while not self.ctl.red_button_event.is_set():
            rclpy.spin_once(self.ctl, timeout_sec=0.1)
        self.ctl.red_button_event.clear()

        self.ctl.get_logger().info(f'red button pressed ...')

    def grab_sequence(self, speed, grip):
        self.ctl.get_logger().info(f'grabbing object ...')

        cmd = f'grab_seq {int(speed)} {int(grip)}'
        self.__send_claw_msg(cmd)

        self.ctl.get_logger().info(f'claw done')

    def open_claw(self):
        self.ctl.get_logger().info(f'releasing object ...')

        cmd = f'open'
        self.__send_claw_msg(cmd)

        self.ctl.get_logger().info(f'claw open')

    def close_claw(self, grip):
        '''
        grip 0-255	
        '''
        self.ctl.get_logger().info(f'closing claw ...')

        cmd = f'close {int(grip)}'
        self.__send_claw_msg(cmd)

        self.ctl.get_logger().info(f'claw closed')

    def claw_up(self, speed):
        '''
        speed 0-255
        '''
        self.ctl.get_logger().info(f'raising claw ...')

        cmd = f'up {int(speed)}'
        self.__send_claw_msg(cmd)

        self.ctl.get_logger().info(f'claw up')


    def claw_down(self, speed):
        self.ctl.get_logger().info(f'claw downwards...')

        cmd = f'down {int(speed)}'
        self.__send_claw_msg(cmd)

        self.ctl.get_logger().info(f'claw down')


    def __send_claw_msg(self, message):
        self.ctl.claw_status_event.clear()

        claw_msg = String()
        claw_msg.data = message
        self.ctl.claw_cmds_publisher.publish(claw_msg)

        #wait to get "task done" message from claw controller
        while not self.ctl.claw_status_event.is_set():
            rclpy.spin_once(self.ctl, timeout_sec=0.5)


    def __del__(self):
        self.ctl.destroy_node()
        rclpy.shutdown()






class RosClawCtl(Node):
    '''ROS2 node for controlling claw controller messages manually'''
    def __init__(self):
        super().__init__('basic_game')
        self.joystick_enable_publisher = self.create_publisher(UInt8, 'joystick/enable', 1)
        self.xcarve_goto_publisher = self.create_publisher(Position, 'xcarve/goto', 1)
        self.claw_cmds_publisher = self.create_publisher(String, 'claw/ctl', 1)

        #coordinates for home position
        self.home_x = 0.0
        self.home_y = 150.0
        #flag to indicate that xcarve is in home position
        self.home_event = threading.Event()
        self.home_event.clear()

        #used as a flag to indicate that a red button pressed message was received
        self.red_button_event = threading.Event()
        self.red_button_event.clear()

        #flag to indicate that a claw status message was received
        self.claw_status_event = threading.Event()
        self.claw_status_event.clear()


        # subscriber for joystick commands
        self.joystick_subscription = self.create_subscription(
            String,
            'joystick/cmd',
            self.joystick_callback,
            1)
        self.joystick_subscription  # prevent unused variable warning 

        # subscriber for claw status message
        self.joystick_subscription = self.create_subscription(
            String,
            'claw/status',
            self.claw_status_callback,
            1)
        self.claw_status_callback  # prevent unused variable warning   

        # subscriber for xcarve current position messages
        self.xcarve_position_subscription = self.create_subscription(
            Position,
            'xcarve/position',
            self.xcarve_position_callback,
            1)
        self.xcarve_position_subscription  # prevent unused variable warning 

        self.last_key = None
        #self.timer = self.create_timer(5, self.timer_callback)
        self.prev_joystick_status = 0

    def claw_status_callback(self, msg):
        if msg.data == 'done':
            self.claw_status_event.set()

    def xcarve_position_callback(self, msg):
        #absolute errors between home and current positions
        max_error = 5.0
        ex = abs(msg.x - self.home_x)
        ey = abs(msg.y - self.home_y)

        #self.get_logger().info(f'position: {ex} {ey}')
        
        # if current position is home position set home flag
        if ex <= max_error and ey <= max_error:
            self.home_event.set()


    def joystick_callback(self, msg):
        if msg.data == 'Button.red':
            self.get_logger().info(f'Publishing: "{msg}"')
            self.red_button_event.set()
            
