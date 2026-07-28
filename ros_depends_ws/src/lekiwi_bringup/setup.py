from glob import glob
from setuptools import find_packages, setup


package_name = "lekiwi_bringup"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="UBRobot",
    maintainer_email="mathzxw2002@gmail.com",
    description="Safe ros2_control bringup and command adaptation for LeKiwi.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cmd_vel_adapter = lekiwi_bringup.cmd_vel_adapter:main",
        ],
    },
)
