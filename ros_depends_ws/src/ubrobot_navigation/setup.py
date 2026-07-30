from setuptools import find_packages, setup


PACKAGE_NAME = "ubrobot_navigation"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubrobot",
    maintainer_email="user@example.com",
    description="Controlled ROS 2 navigation capability for UBRobot.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cmd_vel_guard = ubrobot_navigation.cmd_vel_guard:main",
            "navigate_to_object_server = ubrobot_navigation.navigate_to_object_server:main",
        ],
    },
)
