import os
from glob import glob
from setuptools import setup

package_name = 'ankybot_policy_runner'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'onnxruntime'],
    zip_safe=True,
    maintainer='mingus',
    maintainer_email='you@example.com',
    description='ONNX RL locomotion policy runner for Ankybot quadruped',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_runner = ankybot_policy_runner.policy_runner:main',
        ],
    },
)
