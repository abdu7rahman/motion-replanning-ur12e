from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'reactive_replanning_ur12e'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abdul rahman',
    maintainer_email='mohammedabdulr.1@northeastern.edu',
    description='Reactive motion replanning for UR12e + Robotiq Hand-E via kinematic redundancy',
    license='MIT',
    entry_points={
        'console_scripts': [
            'reactive_replanning = reactive_replanning_ur12e.reactive_replanning:main',
            'scene_setup         = reactive_replanning_ur12e.scene_setup:main',
            'insert_obstacle     = reactive_replanning_ur12e.insert_obstacle:main',
        ],
    },
)
