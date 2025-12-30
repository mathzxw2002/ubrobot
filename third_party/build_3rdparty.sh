# cyclonedds
#git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir -p build install && cd build && 
cmake .. -DCMAKE_INSTALL_PREFIX=../../install -DBUILD_DDSPERF=OFF && 
cmake --build . --target install

export CYCLONEDDS_HOME="./install"

echo $CYCLONEDDS_HOME
