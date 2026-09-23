from glob import glob

from setuptools import setup

package_name = 'formation_gazebo'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Martina Doku',
    maintainer_email='martina.doku.t7@dc.tohoku.ac.jp',
    description='Gazebo backend for the networked formation task.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'leader_node = formation_gazebo.leader_node:main',
            'comm_interface_node = formation_gazebo.comm_interface_node:main',
            'controller_node = formation_gazebo.controller_node:main',
            'evaluation_node = formation_gazebo.evaluation_node:main',
        ],
    },
)
