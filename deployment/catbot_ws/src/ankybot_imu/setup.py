import os
from glob import glob
from setuptools import setup

package_name = 'ankybot_imu'

setup(
    name=package_name,
    version='0.2.3',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # installed (not a console_script) so catbot_bringup can invoke it via `python3 <share-dir path>` as a pre-flight gate
        ('share/' + package_name, ['imu_diag.py']),
    ],
    install_requires=['setuptools', 'adafruit-circuitpython-bno08x', 'adafruit-blinka'],
    zip_safe=True,
    maintainer='mingus',
    maintainer_email='you@example.com',
    description='BNO085 IMU driver, wired directly to the Pi 5 I2C pins, publishing sensor_msgs/Imu',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bno085_publisher = ankybot_imu.bno085_publisher:main',
        ],
    },
)
