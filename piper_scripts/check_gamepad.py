import pygame
import time
import sys

def main():
    """
    一个简单的脚本，用于检测和显示pygame检测到的手柄的轴、按钮和方向键的映射。
    """
    pygame.init()
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("错误：未检测到任何手柄。请确保您的手柄已连接并且驱动正常。")
        pygame.quit()
        sys.exit(1)

    print(f"检测到 {joystick_count} 个手柄。")
    # 默认使用第一个检测到的手柄
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"手柄名称: {joystick.get_name()}")
    print(f"  - 轴 (Axes) 数量: {joystick.get_numaxes()}")
    print(f"  - 按钮 (Buttons) 数量: {joystick.get_numbuttons()}")
    print(f"  - 方向键 (Hats) 数量: {joystick.get_numhats()}")
    print("\n请按下手柄上的按钮或移动摇杆来查看它们的编号...")
    print("按 Ctrl+C 退出程序。")

    # 存储上一次状态，避免重复打印
    last_axis_values = [0.0] * joystick.get_numaxes()
    last_button_states = [0] * joystick.get_numbuttons()
    last_hat_states = [(0, 0)] * joystick.get_numhats()

    try:
        while True:
            pygame.event.pump()  # 处理内部事件

            # --- 检测并打印轴 (Joysticks) ---
            for i in range(joystick.get_numaxes()):
                axis_value = joystick.get_axis(i)
                # 仅当轴的移动幅度大于一个阈值或与上次值显著不同时才打印
                if abs(axis_value) > 0.1 and abs(axis_value - last_axis_values[i]) > 0.05:
                    print(f"轴 (Axis) {i}: {axis_value:.4f}")
                last_axis_values[i] = axis_value

            # --- 检测并打印按钮 (Buttons) ---
            for i in range(joystick.get_numbuttons()):
                button_state = joystick.get_button(i)
                if button_state and not last_button_states[i]:  # 按钮从未按下到按下
                    print(f"按钮 (Button) {i}: 被按下")
                last_button_states[i] = button_state

            # --- 检测并打印方向键 (D-Pad / Hat) ---
            for i in range(joystick.get_numhats()):
                hat_state = joystick.get_hat(i)
                if hat_state != (0, 0) and hat_state != last_hat_states[i]:
                    print(f"方向键 (Hat) {i}: {hat_state}")
                last_hat_states[i] = hat_state

            time.sleep(0.05)  # 降低刷新率，避免刷屏太快

    except KeyboardInterrupt:
        print("\n\n程序已退出。")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
