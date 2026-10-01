from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'keyboard_claw_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', 'keyboard_claw_controller'),
            glob('launch/*.launch.py')
        )
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rosdev',
    maintainer_email='a.pico@posteo.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'keyboard_publisher = keyboard_claw_controller.keyboard_publisher:main',
            'xcarve_controller = keyboard_claw_controller.xcarve_controller:main',
            'ros2_mqtt_bridge = keyboard_claw_controller.ros2_mqtt_bridge:main',
            'basic_game = keyboard_claw_controller.basic_game:main',
            'claw_adapter = keyboard_claw_controller.claw_adapter:main'
        ],
    },
)
