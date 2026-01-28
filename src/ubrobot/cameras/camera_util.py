import logging
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Dict, Optional
from lerobot.cameras.configs import CameraConfig, Cv2Rotation

try:
    import pyrealsense2 as rs
except Exception as e:
    logging.info(f"Could not import realsense: {e}")

from lerobot.cameras import ColorMode
from lerobot.utils.errors import DeviceNotConnectedError

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

logger = logging.getLogger(__name__)


class EnhancedRealSenseCamera(RealSenseCamera):
    def __init__(self, config: RealSenseCameraConfig):
        super().__init__(config)
        
        self.intrinsics: Dict[str, Any] = {
            "color": None,  # 彩色流内参
            "depth": None,  # 深度流内参
            "color_K": None,  # 彩色流内参矩阵（OpenCV格式）
            "depth_K": None,  # 深度流内参矩阵（OpenCV格式）
            "color_dist": None,  # 彩色流畸变系数
            "depth_dist": None,  # 深度流畸变系数
        }
        
        self.latest_rgbd_data: Dict[str, NDArray[Any]] = {
            "color": None,
            "depth": None
        }
        self.rgbd_lock = self.frame_lock  # 复用父类的锁，保证线程安全

    def connect(self, warmup: bool = True) -> None:
        super().connect(warmup=warmup)
        self._load_camera_intrinsics()
        logger.info(f"{self} load intrinsics successful.")

    def _load_camera_intrinsics(self) -> None:
        """
        核心新增功能1：读取并解析相机内参
        - 彩色流内参（fx/fy/ppx/ppy + 内参矩阵 + 畸变系数）
        - 深度流内参（同上）
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} fail to connect, can not get intrinsics.")
        
        if self.rs_profile is None:
            raise RuntimeError(f"{self}: rs_profile not initialized, can not get intrinsics.")

        # 1. 读取彩色流内参
        color_stream = self.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intrin = color_stream.get_intrinsics()
        self.intrinsics["color"] = color_intrin
        # 转换为OpenCV格式内参矩阵 [fx, 0, ppx; 0, fy, ppy; 0, 0, 1]
        self.intrinsics["color_K"] = np.array([
            [color_intrin.fx, 0, color_intrin.ppx],
            [0, color_intrin.fy, color_intrin.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        # 畸变系数 [k1, k2, p1, p2, k3]
        self.intrinsics["color_dist"] = np.array(color_intrin.coeffs, dtype=np.float32)

        # 2. 读取深度流内参（如果启用深度）
        if self.use_depth:
            depth_stream = self.rs_profile.get_stream(rs.stream.depth).as_video_stream_profile()
            depth_intrin = depth_stream.get_intrinsics()
            self.intrinsics["depth"] = depth_intrin
            self.intrinsics["depth_K"] = np.array([
                [depth_intrin.fx, 0, depth_intrin.ppx],
                [0, depth_intrin.fy, depth_intrin.ppy],
                [0, 0, 1]
            ], dtype=np.float32)
            self.intrinsics["depth_dist"] = np.array(depth_intrin.coeffs, dtype=np.float32)

        # 打印内参信息（调试用）
        logger.debug(f"彩色流内参矩阵：\n{self.intrinsics['color_K']}")
        if self.use_depth:
            logger.debug(f"深度流内参矩阵：\n{self.intrinsics['depth_K']}")

        intrin = self.intrinsics["color"]
        print("camera info:\n")
        print(f"fx: {intrin.fx}, fy: {intrin.fy}, ppx: {intrin.ppx}, ppy: {intrin.ppy}, width: {intrin.width}, height: {intrin.height}")

if __name__ == "__main__":
    cfg_param = RealSenseCameraConfig(
        serial_number_or_name="336222070923", # Replace with actual SN
        fps=30,
        width=1280,
        height=720,
        color_mode=ColorMode.BGR, # Request BGR output
        rotation=Cv2Rotation.NO_ROTATION,
        use_depth=True
    )
    rs_camera = EnhancedRealSenseCamera(cfg_param)
    rs_camera.get_intrinsics()