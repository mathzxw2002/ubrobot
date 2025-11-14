#!/bin/bash
# 
#  @file      build.sh
#  @brief     angstrong camera demo program build script for ros2
#
#  Copyright (c) 2023 Angstrong Tech.Co.,Ltd
#
#  @author    Angstrong SDK develop Team
#  @date      2023/03/29
#  @version   1.0
#


# compile ros2 package
# colcon build --symlink-install --cmake-args -DCROSS_COMPILE=aarch64-linux-gnu
# colcon build --symlink-install --cmake-args -DCROSS_COMPILE=arm-linux-gnueabihf
# colcon build --symlink-install
colcon build --symlink-install --cmake-args -DCROSS_COMPILE=x86_64-linux-gnu
