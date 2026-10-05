import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8,String
from claw_machine_msgs.msg import Position
import threading
import time

class ClawCtl():
    




class RosClawCtl(Node):
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
            self.get_logger().info('el don')
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
        '''if msg.data == 'Button.red':
                #when red button pressed , initiate grab sequence
                claw_msg = String()
                #speed , grip
                claw_msg.data = f'open'# {255} {128}'
                self.claw_cmds_publisher.publish(claw_msg)
        '''
            
        
    def timer_callback(self):
        #msg = UInt8()
        #msg.data = 1 if self.prev_joystick_status == 0 else 0
        #self.prev_joystick_status = msg.data

        #self.joystick_enable_publisher.publish(msg)

        msg = Position()
        msg.x = 400.0
        msg.y = 300.0
        #self.xcarve_goto_publisher.publish(msg)

        self.get_logger().info(f'Publishing: "{msg}"')

def main(args=None):
    rclpy.init(args=args)
    ctl = GameCtl()

    #disable joystick
    ctl.get_logger().info(f'disabling joystick...')
    joytick_enable_msg = UInt8()
    joytick_enable_msg.data = 0
    ctl.joystick_enable_publisher.publish(joytick_enable_msg)
    time.sleep(.3)

    #move xcarve to initial position
    ctl.get_logger().info(f'going to home position...')
    xcarve_position_msg = Position()
    xcarve_position_msg.x = 0.0
    xcarve_position_msg.y = 150.0
    ctl.xcarve_goto_publisher.publish(xcarve_position_msg)

    #wait to get to home position
    ctl.home_event.clear()
    while not ctl.home_event.is_set():
        rclpy.spin_once(ctl, timeout_sec=0.5)
    ctl.get_logger().info(f'home position.')

    #enable joystick to send commands
    ctl.get_logger().info(f'enabling joystick ...')
    joytick_enable_msg = UInt8()
    joytick_enable_msg.data = 1
    ctl.joystick_enable_publisher.publish(joytick_enable_msg)


    #wait for red button to be pressed
    while not ctl.red_button_event.is_set():
        rclpy.spin_once(ctl, timeout_sec=0.1)
    ctl.red_button_event.clear()
    ctl.get_logger().info(f'red button pressed ...')

    #disable joystick
    ctl.get_logger().info(f'disabling joystick..')
    joytick_enable_msg = UInt8()
    joytick_enable_msg.data = 0
    ctl.joystick_enable_publisher.publish(joytick_enable_msg)

    ctl.claw_status_event.clear()

    ctl.get_logger().info(f'grabbing object ...')
    claw_msg = String()
    #speed , grip
    claw_msg.data = f'grab_seq {255} {255}'
    ctl.claw_cmds_publisher.publish(claw_msg)

    #wait to get task done message from claw controller
    
    while not ctl.claw_status_event.is_set():
        rclpy.spin_once(ctl, timeout_sec=0.5)
    ctl.get_logger().info(f'claw done')

    time.sleep(1)

    #move xcarve to initial position
    ctl.get_logger().info(f'going to home position..')
    xcarve_position_msg = Position()
    xcarve_position_msg.x = 0.0
    xcarve_position_msg.y = 150.0
    ctl.xcarve_goto_publisher.publish(xcarve_position_msg)

    #wait to get to home position
    ctl.home_event.clear()
    while not ctl.home_event.is_set():
        rclpy.spin_once(ctl, timeout_sec=0.5)
    ctl.get_logger().info(f'home position.')

    time.sleep(2)

    ctl.get_logger().info(f'releasing object ...')
    claw_msg = String()
    claw_msg.data = f'open'
    ctl.claw_cmds_publisher.publish(claw_msg)

    time.sleep(2)



    #rclpy.spin(ctl)

    ctl.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
