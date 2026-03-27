import torch
import cv2
import numpy as np
import mss
import serial
import serial.tools.list_ports
import threading
import time
import json
import os
import sys
from collections import deque

# 虚拟手柄库
import vgamepad

# 导入模型结构 (假设 models 文件夹在同级目录)
from models.cnn_lstm import CNNLSTM


# ============== 配置加载 ==============
def load_config():
    config_path = 'config.json'
    if not os.path.exists(config_path):
        print(f"✗ 错误: 缺少配置文件 {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


cfg = load_config()

# ============== 全局变量 ==============
running = True
latest_packet = None  # 原始串口数据包
latest_ai_controls = [0.0, 0.0, 0.0, -1.0]  # AI输出 [roll, pitch, yaw, throttle]
data_lock = threading.Lock()


# ============== 串口解析逻辑 ==============
def parse_channel(byte_low, byte_high):
    """
    解析单个通道值
    公式: num = (low+1) + high*256
    范围: 0~2048 -> 映射到 [-1, 1]
    """
    low = byte_low
    high = byte_high
    num = (low + 1) + (high * 256)
    val = 2.0 * (num / 2048.0) - 1.0
    return round(val, 4)


# ============== 线程 1: 串口监听 ==============
def serial_thread():
    global latest_packet, running
    ser = None

    # 自动检测串口 (如果配置的不对)
    ports = [p.device for p in serial.tools.list_ports.comports()]
    target_port = cfg['serial_port']

    if target_port not in ports:
        print(f"⚠ 警告: 配置的串口 {target_port} 未找到，尝试自动连接...")
        if ports:
            target_port = ports[0]
            print(f"→ 已自动连接到 {target_port}")
        else:
            print("✗ 错误: 找不到任何串口设备！")
            return

    try:
        ser = serial.Serial(target_port, cfg['baudrate'], timeout=0.01)
        print(f"✓ 串口已连接: {target_port}")
        buffer = bytearray()

        while running:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                buffer.extend(data)

                # 解析 16 字节包
                while len(buffer) >= 16:
                    # 简单的帧头检测可以加在这里，如果协议有的话
                    # 目前假设数据流是完美的 16 字节对齐
                    packet = buffer[:16]
                    buffer = buffer[16:]

                    with data_lock:
                        latest_packet = packet
            else:
                time.sleep(0.001)
    except Exception as e:
        print(f"✗ 串口异常: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()


# ============== 线程 2: AI 推理 ==============
# ============== 线程 2: AI 推理 ==============
def ai_thread():
    global running, latest_ai_controls

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 使用设备: {device}")

    # 1. 加载掩膜
    mask = None
    mask_path = cfg.get('mask_path')
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask.astype(np.float32) / 255.0
            print(f"✓ 掩膜已加载: {mask_path}, 尺寸: {mask.shape}")
        else:
            print(f"✗ 无法读取掩膜文件: {mask_path}")
    else:
        print("⚠ 未配置掩膜或文件不存在，跳过掩膜处理")

    # 2. 初始化模型
    model = CNNLSTM(
        input_shape=(3, cfg['img_size'], cfg['img_size']),
        lstm_hidden=cfg['lstm_hidden'],
        lstm_layers=cfg['lstm_layers'],
        output_dim=4
    ).to(device)

    # 3. 加载权重
    model_path = cfg['model_path']
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ 模型权重已加载: {model_path}")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            running = False
            return
    else:
        print(f"✗ 找不到模型文件: {model_path}")
        running = False
        return

    model.eval()

    # 4. 帧缓冲区初始化
    frame_buffer = deque(maxlen=cfg['seq_len'])
    dummy_frame = np.zeros((cfg['img_size'], cfg['img_size'], 3), dtype=np.float32)
    for _ in range(cfg['seq_len']):
        frame_buffer.append(dummy_frame)

    # 5. 截屏循环
    with mss.mss() as sct:
        monitor = sct.monitors[1]

        while running:
            t_start = time.time()

            # A. 截屏
            screenshot = np.array(sct.grab(monitor))

            # B. 预处理
            # 1) 转换颜色 BGRA -> BGR
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            # 2) 【关键修改】转黑白
            # 先转为单通道灰度
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 再转回 3 通道 (R=G=B)
            # 这一步是为了适配你模型的 Conv2d(3, ...) 结构
            # 如果不转，模型会报错。此时图片视觉上依然是黑白的。
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # 3) 应用掩膜
            if mask is not None:
                mask_h, mask_w = mask.shape
                frame_resized = cv2.resize(frame, (mask_w, mask_h))
                mask_3ch = np.stack([mask, mask, mask], axis=-1)
                frame_masked = frame_resized * mask_3ch
                frame = cv2.resize(frame_masked, (cfg['img_size'], cfg['img_size']))
            else:
                frame = cv2.resize(frame, (cfg['img_size'], cfg['img_size']))

            # 4) 归一化
            frame = frame.astype(np.float32) / 255.0

            # C. 加入缓冲区
            frame_buffer.append(frame)

            # D. 构建 Tensor
            sequence = np.array(list(frame_buffer))
            tensor = torch.from_numpy(sequence).permute(0, 3, 1, 2)
            tensor = tensor.unsqueeze(0).to(device)

            # E. 推理
            with torch.no_grad():
                output = model(tensor)

            # F. 后处理
            controls = output[0].cpu().numpy()
            controls[3] = controls[3] * 2 - 1

            with data_lock:
                latest_ai_controls = controls.tolist()

            # G. 帧率控制
            elapsed = time.time() - t_start
            sleep_time = (1.0 / cfg['inference_fps']) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ============== 主循环 ==============
# ============== 主循环 ==============
def main():
    global running, latest_packet, latest_ai_controls

    print("\n▶ 正在初始化系统...")

    # 初始化虚拟手柄
    try:
        gamepad = vgamepad.VX360Gamepad()
        print("✓ 虚拟手柄已创建")
    except Exception as e:
        print(f"✗ 创建手柄失败: {e}")
        print("  请确认已安装 ViGEmBus 驱动")
        return

    # 启动线程
    t_serial = threading.Thread(target=serial_thread, daemon=True)
    t_ai = threading.Thread(target=ai_thread, daemon=True)
    t_serial.start()
    t_ai.start()

    print("▶ 系统运行中... (按 Ctrl+C 退出)")

    current_mode = "MANUAL"

    while running:
        # 1. 获取遥控器数据
        remote_controls = [0, 0, 0, -1]
        mode_switch = "manual"
        ch7 = 0.0

        with data_lock:
            packet = latest_packet
            ai_data = latest_ai_controls.copy()

        if packet:
            # 解析通道
            ch1 = parse_channel(packet[0], packet[1])
            ch2 = parse_channel(packet[2], packet[3])
            ch3 = parse_channel(packet[4], packet[5])
            ch4 = parse_channel(packet[6], packet[7])
            ch7 = parse_channel(packet[12], packet[13])

            # 映射到统一格式: [Roll, Pitch, Yaw, Throttle]
            remote_controls = [ch1, ch2, ch4, ch3]

            # 模式判断
            if ch7 < -0.5:
                mode_switch = "auto"

        # 2. 数据源选择
        if mode_switch == "auto":
            final_controls = ai_data
            current_mode = "AUTO"
        else:
            final_controls = remote_controls
            current_mode = "MANUAL"

        # 3. 映射到虚拟手柄
        roll = np.clip(final_controls[0], -1.0, 1.0)
        pitch = np.clip(final_controls[1], -1.0, 1.0)
        yaw = np.clip(final_controls[2], -1.0, 1.0)
        throttle = np.clip(final_controls[3], -1.0, 1.0)

        # --- 修改部分开始 ---

        # A. 油门 -> 左扳机 (转换范围: -1~1 -> 0~1)
        trigger_val = (float(throttle) + 1.0) / 2.0
        gamepad.left_trigger_float(trigger_val)

        # B. 左摇杆: X=偏航, Y=0.0 (不再控制油门，强制归零)
        gamepad.left_joystick_float(float(yaw), 0.0)

        # C. 右摇杆: X=横滚, Y=俯仰 (保持不变)
        gamepad.right_joystick_float(float(roll), float(pitch))

        # --- 修改部分结束 ---

        gamepad.update()

        # 调试打印
        if cfg['debug']:
            print(f"\rMode: {current_mode} | "
                  f"R:{roll:+.2f} P:{pitch:+.2f} Y:{yaw:+.2f} T:{throttle:+.2f} | "
                  f"CH7:{ch7:+.2f}", end="")

        time.sleep(0.005)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        running = False
        print("\n程序已退出。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        running = False
        print("\n程序已退出。")
