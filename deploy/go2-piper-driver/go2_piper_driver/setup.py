from setuptools import find_packages, setup

package_name = "go2_piper_driver"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["launch/go2_piper_bringup.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubrobot",
    maintainer_email="ubrobot@example.com",
    description=(
        "Go2+Piper hardware driver container: guarded /cmd_vel -> Go2 Unitree "
        "DDS body, and Piper arm CAN commands (JointCtrl/GripperCtrl)."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "go2_bridge_node = go2_piper_driver.go2_bridge_node:main",
            "piper_driver_node = go2_piper_driver.piper_driver_node:main",
        ],
    },
)
