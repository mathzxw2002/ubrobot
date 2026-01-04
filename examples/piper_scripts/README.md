

https://github.com/agilexrobotics/piper_sdk



bash find_all_can_port.sh

bash can_activate.sh can0 1000000

python piper_disable.py 


https://github.com/agilexrobotics/handeye_calibration_ros


Recognize Pick (OpenVINO Grasp Detection + OpenVINO Object Segmentation)
https://intel.github.io/ros2_grasp_library/docs/doc/recognize_pick.html



https://www.hackster.io/agilexrobotics/real-time-6d-pose-generation-grasp-planning-toolkit-3c9af6




## 

```
/home/unitree/piper_ws/src/GraspGen/piper_kinematics/src/piper_ik_node.cpp:15:10: fatal error: piper_msgs/PosCmd.h: No such file or directory
   15 | #include <piper_msgs/PosCmd.h>
      |          ^~~~~~~~~~~~~~~~~~~~~
compilation terminated.
[  2%] Built target bond_generate_messages_py
[  2%] Built target diagnostic_msgs_generate_messages_cpp
make[2]: *** [GraspGen/piper_kinematics/CMakeFiles/piper_ik_node.dir/build.make:76: GraspGen/piper_kinematics/CMakeFiles/piper_ik_node.dir/src/piper_ik_node.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:3482: GraspGen/piper_kinematics/CMakeFiles/piper_ik_node.dir/all] Error 2
make[1]: *** Waiting for unfinished jobs....
[  2%] Built target diagnostic_msgs_generate_messages_eus
/home/unitree/piper_ws/src/GraspGen/piper_kinematics/src/piper_ik_node_use_yaik.cpp:13:10: fatal error: piper_msgs/PosCmd.h: No such file or directory
   13 | #include <piper_msgs/PosCmd.h>
      |          ^~~~~~~~~~~~~~~~~~~~~
compilation terminated.
make[2]: *** [GraspGen/piper_kinematics/CMakeFiles/piper_ik_node_use_yaik.dir/build.make:76: GraspGen/piper_kinematics/CMakeFiles/piper_ik_node_use_yaik.dir/src/piper_ik_node_use_yaik.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:3508: GraspGen/piper_kinematics/CMakeFiles/piper_ik_node_use_yaik.dir/all] Error 2
make: *** [Makefile:146: all] Error 2
Invoking "make -j4 -l4" failed
```

catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DCATKIN_WHITELIST_PACKAGES="piper_msgs"

