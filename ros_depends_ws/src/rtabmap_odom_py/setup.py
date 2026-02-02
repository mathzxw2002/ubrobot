from setuptools import setup, find_packages
import os
import subprocess

# 编译pybind11模块（自动执行cmake）
def build_extension():
    # 创建build目录
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    os.makedirs(build_dir, exist_ok=True)
    
    # 执行cmake和make
    subprocess.check_call(
        ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir
    )
    subprocess.check_call(["make", "-j4"], cwd=build_dir)

build_extension()

# 安装配置
setup(
    name="rtabmap_odom_py",  # 包名
    version="0.1.0",
    description="UBRobot RealSense Odometry Module",
    packages=find_packages(),  # 自动识别rtabmap_odom_py和rtabmap_odom_py.odom
    package_data={
        "rtabmap_odom_py.odom": ["rs_odom_module.so"],  # 包含编译后的.so文件
    },
    include_package_data=True,
    data_files=[
        ("rtabmap_odom_py/odom", ["odom/rs_odom_module.so"]),
    ],
    zip_safe=False,  # 必须设为False，因为.so文件不能被压缩
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0"
    ],
    author="Xiaowei Zhao",
    author_email="mathzxw2002@gmail.com",
    url="https://your-repo.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Linux",
    ],
)
