import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'cam_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install the launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Install the config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Camera testing and info publishing package',
    license='Apache-2.0',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # executable_name = package_name.script_name:main_function
            'camera_tester = cam_test.camera_testing:main',
            'camera_info_pub = cam_test.camInfo:main',
        ],
    },
)
