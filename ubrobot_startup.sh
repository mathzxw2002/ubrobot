source /opt/ros/jazzy/setup.bash

export DASHSCOPE_API_KEY="sk-479fdd23120c4201bff35a107883c7c3"
# 配置半精度模式（布尔值建议用小写或引号包裹，避免Shell解析问题）
export IS_HALF="True"

#source /home/china/vision_opencv/install/setup.bash
#sudo chmod 666 /dev/ttyACM0
python3 src/chat_ui/app.py
#python test.py
