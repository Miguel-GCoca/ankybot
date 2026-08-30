import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'speech_detection'

setup(
    name=package_name,
    version='0.4.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mingus',
    maintainer_email='mingus@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'speech_2_txt = speech_detection.speech_recognizer_node:main',
            'command_interpreter = speech_detection.command_interpreter:main',
            'dino_head_controller = speech_detection.dino_head_controller:main',
        ],
    },
)
