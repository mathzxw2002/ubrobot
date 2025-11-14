from setuptools import setup

package_name = 'yahboomcar_visual'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nx-ros2',
    maintainer_email='123456789@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'simple_AR = yahboomcar_visual.simple_AR:main',
        'ascam_rgb_image = yahboomcar_visual.ascam_rgb_image:main',
        'ascam_depth_image = yahboomcar_visual.ascam_depth_image:main',
        'QRcode_Parsing = yahboomcar_visual.QRcode_Parsing:main'
        ],
    },
)
