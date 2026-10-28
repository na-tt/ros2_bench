from setuptools import find_packages, setup

package_name = 'yolocar_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/bench.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nat',
    maintainer_email='panatt17@gmail.com',
    description=(
        'Four-node ROS 2 chain for the yolocar pipeline: acquisition -> '
        'preprocessing -> inference -> postprocessing.'
    ),
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'acquisition = yolocar_ros.acquisition_node:main',
            'preprocessing = yolocar_ros.preprocessing_node:main',
            'inference = yolocar_ros.inference_node:main',
            'postprocessing = yolocar_ros.postprocessing_node:main',
        ],
    },
)
