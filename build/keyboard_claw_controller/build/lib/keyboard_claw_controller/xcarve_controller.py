#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time
import threading
from claw_machine_msgs.msg import Position
import re


class XcarveController(Node):
    def __init__(self):
        super().__init__('xcarve_controller')

        self.last_received_cmd = 'stop'

        self.lock = threading.Lock()

        # subscriber for movement commands for xcarve
        self.subscription = self.create_subscription(
            String,
            'arrow_key',
            self.movement_cmd_callback,
            1)
        self.subscription  # prevent unused variable warning

        self.serial_port = serial.Serial("/dev/xcarve", 115200)
        #self.serial_port = serial.Serial("/dev/ttyUSB0", 115200)
        self.serial_port.parity = serial.PARITY_NONE  # Parity. Options include PARITY_NONE, PARITY_EVEN, PARITY_ODD
        self.serial_port.stopbits = serial.STOPBITS_ONE  # Stop bits. Options include STOPBITS_ONE, STOPBITS_ONE_POINT_FIVE, STOPBITS_TWO
        self.serial_port.bytesize = serial.EIGHTBITS  # Data bits. Options include FIVEBITS, SIXBITS, SEVENBITS, EIGHTBITS
        self.serial_port.timeout = 10  # Read timeout in seconds (None for blocking mode, 0 for non-blocking mode)

        # sleep after connecting serial port to xcarve. it needs some seconds to start
        time.sleep(5)
        self.homing()

        # publisher for xcarve current position
        self.xcarve_position_publisher = self.create_publisher(Position, 'xcarve_position', 1)
        self.timer_publish_position = self.create_timer(0.1, self.timer_publish_position_callback)

    def xcarve_stop_cmd(self):
        # Send the jog cancel command (0x85)
        self.lock.acquire()
        self.serial_port.flushInput()
        self.serial_port.write(b'\x85')
        #ok_string =  self.serial_port.readline().decode('utf-8')
        self.lock.release()

    def movement_cmd_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

        # when receives a move command it tells the xcarve to move all the way
        # and keeps it while it receives the same comand continously
        # if no command is sent in a while, it stops xcarve

        # true if previous command received is the same as the new received commad
        previous_received_same_flag = True

        #if last cmd was different,sent stop and flush buffer cmd to xcarve first
        if self.last_received_cmd != msg.data:
            self.xcarve_stop_cmd()
            previous_received_same_flag = False

        if msg.data == "Key.up":
            # move forward
            # G91 = incremental distances , G21 = millimeters
            # G90 = absolute coordinates
            #self.send_cmd("$J=G91 G21 X40 F8000")

            #send cmd only if previous was differnet
            if not previous_received_same_flag: 
                self.send_cmd("$J=G90 G21 Y780 F8000")
        elif msg.data == "Key.down":
            # move backward
            #self.send_cmd("$J=G91 G21 X-40 F8000")
            if not previous_received_same_flag: 
                self.send_cmd("$J=G90 G21 Y0 F8000")
        elif msg.data == "Key.right":
            # move right
            #self.send_cmd("$J=G91 G21 Y40 F8000")
            if not previous_received_same_flag: 
                self.send_cmd("$J=G90 G21 X0 F8000")
        elif msg.data == "Key.left":
            # move left
            #self.send_cmd("$J=G91 G21 Y-40 F8000")
            if not previous_received_same_flag: 
                self.send_cmd("$J=G90 G21 X780 F8000")
        elif msg.data == "Key.stop":
            if not previous_received_same_flag: 
                self.xcarve_stop_cmd()

        self.last_received_cmd = msg.data
        


    def timer_publish_position_callback(self):  
        pos_msg = self.read_position()
        self.xcarve_position_publisher.publish(pos_msg)
        #self.get_logger().info(f'{pos_msg}')


    def homing(self):
        self.serial_port.flushInput()
        #print('Going home')
        self.get_logger().info('Going home')
        output = self.send_cmd("$H")
        self.get_logger().info(output)
        if output.find('ALARM', 0, len(output)) != -1:
            #print ('There was an alarm lock. Killing it')
            self.get_logger().info('There was an alarm lock. Killing it')
            #read and discard next four lines (part of error message)
            for i in range(4):
                self.serial_port.readline().decode('utf-8')
            self.send_cmd("$X")
        #self.send_cmd("g91") # sending again relative positioning
        self.send_cmd("g21") # use millimiter units (g20 for inches)
        #self.send_cmd("$10=14")
        self.get_logger().info('home position')

    def send_cmd(self,cmd):
        self.get_logger().info(f'{cmd}')
        b_cmd = cmd+'\n'
        self.lock.acquire()
        self.serial_port.flushInput()
        self.serial_port.write(b_cmd.encode('utf-8'))
        grbl_out = self.serial_port.readline().decode('utf-8')
        self.lock.release()
        self.get_logger().info('send_cmd : ' + cmd + ": " + grbl_out.strip())
        return grbl_out
    

    def get_raw_position_xcarve(self):
        # send status command and read two lines (first one is an "ok" and second one info message)
        self.lock.acquire()
        self.serial_port.flushInput()
        self.serial_port.write("?\n".encode('utf-8'))
        status_string  = self.serial_port.readline().decode('utf-8')
        ok_string =  self.serial_port.readline().decode('utf-8')
        self.lock.release()
        return status_string


    def read_position(self):
        mess = Position() # current position message
        mess.header.stamp = self.get_clock().now().to_msg()
        # send status request to xcarve
        grbl_out = self.get_raw_position_xcarve()
        # parse request 
        # return message is in the form:
        # "<Idle|WPos:0.000,0.000,0.000|Bf:15,127|FS:0,0|WCO:-784.000,-784.000,-1.000>"
        # extract wpos from message (three floating point numbers)
        try:
            str_position = re.search(r'WPos:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)',grbl_out).groups()
            mess.x = float(str_position[0])
            mess.y = float(str_position[1])
            mess.z = float(str_position[2])
        except :
            self.get_logger().info(f'Error reading position, msg:{grbl_out}')
            mess.x = -1.0
            mess.y = -1.0
            mess.z = -1.0
        
        return mess

def main(args=None):
    rclpy.init(args=args)
    xcarve_controller = XcarveController()
    rclpy.spin(xcarve_controller)
    xcarve_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()







'''
class XcarveController(Node):
    def __init__(self):
        super().__init__('xcarve_controller')

        self.last_received_cmd = 'stop'

        self.lock = threading.Lock()

        # subscriber for movement commands for xcarve
        self.subscription = self.create_subscription(
            String,
            'arrow_key',
            self.movement_cmd_callback,
            1)
        self.subscription  # prevent unused variable warning


        self.serial_port = serial.Serial("/dev/ttyUSB0", 115200)
        self.serial_port.parity = serial.PARITY_NONE  # Parity. Options include PARITY_NONE, PARITY_EVEN, PARITY_ODD
        self.serial_port.stopbits = serial.STOPBITS_ONE  # Stop bits. Options include STOPBITS_ONE, STOPBITS_ONE_POINT_FIVE, STOPBITS_TWO
        self.serial_port.bytesize = serial.EIGHTBITS  # Data bits. Options include FIVEBITS, SIXBITS, SEVENBITS, EIGHTBITS
        self.serial_port.timeout = 10  # Read timeout in seconds (None for blocking mode, 0 for non-blocking mode)

        # sleep after connecting serial port to xcarve. it needs some seconds to start
        time.sleep(5)
        self.homing()

        # publisher for xcarve current position
        self.xcarve_position_publisher = self.create_publisher(Position, 'xcarve_position', 1)
        self.timer_publish_position = self.create_timer(0.1, self.timer_publish_position_callback)

        # watchdog timer, if it does not get any movement command within n counts, stops xcarve
        self.watchdog_counter = 0
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)

    def xcarve_stop_cmd(self):
        # Send the jog cancel command (0x85)
        self.lock.acquire()
        self.serial_port.flushInput()
        self.serial_port.write(b'\x85')
        #ok_string =  self.serial_port.readline().decode('utf-8')
        self.lock.release()

    def watchdog_callback(self):
        self.watchdog_counter += 1
        #self.get_logger().info("entro watchdog")
        if self.watchdog_counter >= 2:
            #?self.get_logger().info("watchdog vencido")
            self.xcarve_stop_cmd()
            self.last_received_cmd = "stop" 
            self.watchdog_counter = 0

    def movement_cmd_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

        # when receives a move command it tells the xcarve to move all the way
        # and keeps it while it receives the same comand continously
        # if no command is sent in a while, it stops xcarve

        # true if previous command received is the same as the new received commad
        previous_received_same_flag = True

        #if last cmd was different,sent stop and flush buffer cmd to xcarve first
        if self.last_received_cmd != msg.data:
            self.xcarve_stop_cmd()
            previous_received_same_flag = False

        if msg.data == "Key.up":
            # move forward
            # G91 = incremental distances , G21 = millimeters
            # G90 = absolute coordinates
            #self.send_cmd("$J=G91 G21 X40 F8000")

            #send cmd only if previous was differnet
            if not previous_received_same_flag: 
                self.send_cmd("$J=G90 G21 Y780 F8000")
            self.watchdog_counter = 0
        elif msg.data == "Key.down":
            # move backward
            #self.send_cmd("$J=G91 G21 X-40 F8000")
            if not previous_received_same_flag: 
                self.send_cmd("$J=G90 G21 Y0 F8000")
            self.watchdog_counter = 0
        elif msg.data == "Key.right":
            # move right
            #self.send_cmd("$J=G91 G21 Y40 F8000")
            if not previous_received_same_flag: 
                self.send_cmd("$J=G90 G21 X0 F8000")
            self.watchdog_counter = 0
        elif msg.data == "Key.left":
            # move left
            #self.send_cmd("$J=G91 G21 Y-40 F8000")
            if not previous_received_same_flag: 
                self.send_cmd("$J=G90 G21 X780 F8000")
            self.watchdog_counter = 0

        self.last_received_cmd = msg.data
        


    def timer_publish_position_callback(self):  
        pos_msg = self.read_position()
        self.xcarve_position_publisher.publish(pos_msg)
        #self.get_logger().info(f'{pos_msg}')


    def homing(self):
        self.serial_port.flushInput()
        #print('Going home')
        self.get_logger().info('Going home')
        output = self.send_cmd("$H")
        self.get_logger().info(output)
        if output.find('ALARM', 0, len(output)) != -1:
            #print ('There was an alarm lock. Killing it')
            self.get_logger().info('There was an alarm lock. Killing it')
            #read and discard next four lines (part of error message)
            for i in range(4):
                self.serial_port.readline().decode('utf-8')
            self.send_cmd("$X")
        #self.send_cmd("g91") # sending again relative positioning
        self.send_cmd("g21") # use millimiter units (g20 for inches)
        #self.send_cmd("$10=14")
        self.get_logger().info('home position')

    def send_cmd(self,cmd):
        self.get_logger().info(f'{cmd}')
        b_cmd = cmd+'\n'
        self.lock.acquire()
        self.serial_port.flushInput()
        self.serial_port.write(b_cmd.encode('utf-8'))
        grbl_out = self.serial_port.readline().decode('utf-8')
        self.lock.release()
        self.get_logger().info('send_cmd : ' + cmd + ": " + grbl_out.strip())
        return grbl_out
    

    def get_raw_position_xcarve(self):
        # send status command and read two lines (first one is an "ok" and second one info message)
        self.lock.acquire()
        self.serial_port.flushInput()
        self.serial_port.write("?\n".encode('utf-8'))
        status_string  = self.serial_port.readline().decode('utf-8')
        ok_string =  self.serial_port.readline().decode('utf-8')
        self.lock.release()
        return status_string


    def read_position(self):
        mess = Position() # current position message
        mess.header.stamp = self.get_clock().now().to_msg()
        # send status request to xcarve
        grbl_out = self.get_raw_position_xcarve()
        # parse request 
        # return message is in the form:
        # "<Idle|WPos:0.000,0.000,0.000|Bf:15,127|FS:0,0|WCO:-784.000,-784.000,-1.000>"
        # extract wpos from message (three floating point numbers)
        try:
            str_position = re.search(r'WPos:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)',grbl_out).groups()
            mess.x = float(str_position[0])
            mess.y = float(str_position[1])
            mess.z = float(str_position[2])
        except :
            self.get_logger().info(f'Error reading position, msg:{grbl_out}')
            mess.x = -1.0
            mess.y = -1.0
            mess.z = -1.0
        
        return mess

def main(args=None):
    rclpy.init(args=args)
    xcarve_controller = XcarveController()
    rclpy.spin(xcarve_controller)
    xcarve_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
'''
