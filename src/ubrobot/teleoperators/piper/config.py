
from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

@TeleoperatorConfig.register_subclass("piper")
@dataclass
class PiperTeleoperatorConfig(TeleoperatorConfig):
    pass

@TeleoperatorConfig.register_subclass("piper_keyboard")
@dataclass
class PiperKeyboardTeleopConfig(KeyboardTeleopConfig):
    pass
