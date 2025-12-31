sudo chmod 666 /dev/ttyACM0
python  -m  src.ubrobot.robots.so101_follower.so101_host --robot.id=my_so101 --robot.port=/dev/ttyACM0
