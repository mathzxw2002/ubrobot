from setuptools import find_packages, setup

package_name = "go2_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["launch/go2_bringup.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubrobot",
    maintainer_email="ubrobot@example.com",
    description="Go2 ROS 2 bridge from guarded /cmd_vel to the Unitree DDS body",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "go2_bridge_node = go2_bridge.go2_bridge_node:main",
        ],
    },
)
