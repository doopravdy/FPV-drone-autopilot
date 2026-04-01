import serial
import threading
import time
import cv2
import numpy as np
import mss
import os
import json
import re
import csv

# ============== 配置参数 ==============
SERIAL_PORT = 'COM7'
BAUD_RATE = 115200
VIDEO_FPS = 20
CSV_FILENAME = 'telemetry.csv'
CONFIG_FILE = 'config.json'
LOG_FILE = 'check_log.txt'

# ============== 全局变量 ==============
recording_flag = threading.Event()
running = True
latest_data = ""
data_lock = threading.Lock()

# 录制状态资源
current_flight_index = 1
csv_file = None
csv_writer = None
frame_counter = 0


# ============== 启动检查与修复逻辑 (优化版) ==============
def check_and_repair_folders():
    """
    扫描文件夹，仅在发现异常(断层/配置错误)时才写入日志。
    """
    # 1. 扫描现有文件夹 (内存操作)
    all_items = os.listdir('.')
    flight_pattern = re.compile(r'^flight(\d+)$')
    matched_folders = []

    for item in all_items:
        if os.path.isdir(item):
            match = flight_pattern.match(item)
            if match:
                index = int(match.group(1))
                matched_folders.append((index, item))

    matched_folders.sort(key=lambda x: x[0])

    # 2. 计算状态
    max_index = 0
    gap_list = []  # 记录断层位置

    if not matched_folders:
        max_index = 0
    else:
        max_index = matched_folders[-1][0]
        # 检查断层
        expected_index = 1
        for idx, folder_name in matched_folders:
            if idx != expected_index:
                gap_list.append(f"位置 flight{expected_index} 缺失 (当前为 {folder_name})")
            expected_index = idx + 1

    # 3. 检查配置文件
    config_needs_update = False
    json_index = 0
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                json_index = config.get('last_index', 0)
        except:
            json_index = 0

    if json_index != max_index:
        config_needs_update = True

    # 4. 判断是否需要修复和记录日志
    need_repair = len(gap_list) > 0
    need_log = need_repair or config_needs_update

    # --- 分支A: 一切正常，直接返回 ---
    if not need_log:
        print("✓ 自检正常，文件夹序列连续。")
        return max_index + 1

    # --- 分支B: 发现异常，处理日志和修复 ---
    try:
        # 只有在异常时才创建/打开日志文件
        with open(LOG_FILE, 'w', encoding='utf-8') as log_f:
            log_f.write("=" * 40 + "\n")
            log_f.write("启动自检程序 - 发现异常\n")
            log_f.write("=" * 40 + "\n")

            print("⚠ 检测到异常，正在处理... (详情见 check_log.txt)")

            # 处理断层修复
            if need_repair:
                log_f.write(f"检测到断层: {gap_list}\n")
                log_f.write("正在执行自动修复 (重排序)...\n")

                new_index = 1
                for old_idx, old_name in matched_folders:
                    new_name = f"flight{new_index}"
                    if old_name != new_name:
                        try:
                            os.rename(old_name, new_name)
                            log_f.write(f"  修复：{old_name} -> {new_name}\n")
                        except Exception as e:
                            log_f.write(f"  错误：无法重命名 {old_name}，{e}\n")
                    new_index += 1

                max_index = new_index - 1
                log_f.write("修复完成。\n")
                print("  -> 文件夹断层已修复")

            # 处理配置文件更新
            if config_needs_update:
                log_f.write(f"配置文件记录({json_index})与实际({max_index})不一致，修正中...\n")
                try:
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump({'last_index': max_index}, f)
                    log_f.write("配置文件已更新。\n")
                    print("  -> 配置文件已同步")
                except Exception as e:
                    log_f.write(f"配置更新失败: {e}\n")

            log_f.write("=" * 40 + "\n")

    except Exception as e:
        print(f"日志写入失败: {e}")

    return max_index + 1


# ============== 录制会话管理 ==============
def start_recording_session(index):
    """开始录制：创建文件夹、打开CSV"""
    global csv_file, csv_writer, frame_counter

    folder_name = f"flight{index}"
    image_folder = os.path.join(folder_name, 'images')

    try:
        os.makedirs(image_folder)
        print(f"▶ [系统] 创建录制目录: {folder_name}/")

        csv_path = os.path.join(folder_name, CSV_FILENAME)
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)

        frame_counter = 0
        print(f"▶ [系统) 数据写入就绪")
    except Exception as e:
        print(f"✗ [错误] 创建失败: {e}")


def stop_recording_session(index):
    """结束录制：关闭CSV、更新配置"""
    global csv_file, csv_writer

    if csv_file:
        csv_file.close()
        print(f"■ [系统] CSV已保存")

    csv_file = None
    csv_writer = None

    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({'last_index': index}, f)
        print(f"■ [系统] 记录已更新: last_index = {index}")
    except Exception as e:
        print(f"✗ [错误] 配置更新失败: {e}")


# ============== 数据处理核心 ==============
def process_hex_data(hex_values, frame_index):
    """
    数据处理逻辑：
    1. 时间 = 帧/20
    2. num = (low+1) + high*256
    3. val = 2*(num/2048) - 1
    4. 交换最后两列
    5. 末列映射 [-1,1] -> [0,1]
    """
    time_seconds = frame_index / 20.0
    calculated_values = []

    for i in range(0, len(hex_values), 2):
        if i + 1 < len(hex_values):
            try:
                byte_low = int(hex_values[i], 16)
                byte_high = int(hex_values[i + 1], 16)

                # 核心公式
                num = (byte_low + 1) + (byte_high * 256)
                value = 2.0 * (num / 2048.0) - 1.0

                calculated_values.append(round(value, 2))
            except:
                calculated_values.append(0.0)

    # 交换最后两列
    if len(calculated_values) >= 2:
        calculated_values[-1], calculated_values[-2] = calculated_values[-2], calculated_values[-1]

    # 末列归一化
    if len(calculated_values) >= 1:
        val_mapped = (calculated_values[-1] + 1) / 2.0
        calculated_values[-1] = round(val_mapped, 2)

    return [str(time_seconds)] + [str(v) for v in calculated_values]


# ============== 线程函数 ==============
def serial_reader():
    """串口监听线程"""
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.001)
        print(f"✓ 串口 {SERIAL_PORT} 已连接")
        buffer = bytearray()
        PACKET_SIZE = 16

        while running:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                buffer.extend(data)

                while len(buffer) >= PACKET_SIZE:
                    packet = buffer[:PACKET_SIZE]
                    buffer = buffer[PACKET_SIZE:]
                    tail = packet[-2:]
                    payload = packet[:8]

                    with data_lock:
                        global latest_data
                        latest_data = ' '.join(f'{b:02X}' for b in payload)

                    # 触发检测
                    if tail == b'\xFF\x07':
                        if not recording_flag.is_set():
                            print(f"\n▶ [触发] 收到开始信号 FF 07")
                            recording_flag.set()
                    elif tail == b'\x00\x00':
                        if recording_flag.is_set():
                            print(f"\n■ [触发] 收到结束信号 00 00")
                            recording_flag.clear()

            time.sleep(0.001)
    except Exception as e:
        print(f"✗ 串口错误: {e}")
    finally:
        if ser: ser.close()


def main_loop():
    """主循环"""
    global current_flight_index, frame_counter, csv_writer

    # 启动时检查
    current_flight_index = check_and_repair_folders()

    print("\n系统就绪，等待触发信号...")

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        last_time = time.time()
        interval = 1.0 / VIDEO_FPS
        prev_recording_state = False

        while running:
            current_time = time.time()
            is_recording = recording_flag.is_set()

            # --- 状态机逻辑 ---

            # 开始录制 (上升沿)
            if is_recording and not prev_recording_state:
                start_recording_session(current_flight_index)

            # 停止录制 (下降沿)
            if not is_recording and prev_recording_state:
                stop_recording_session(current_flight_index)
                current_flight_index += 1
                print(f"✓ 就绪。下一录制序号: flight{current_flight_index}")

            # --- 录制核心 ---
            if is_recording and (current_time - last_time >= interval):
                try:
                    # 截图
                    screenshot = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                    img_name = f"frame_{frame_counter:06d}.jpg"
                    img_path = os.path.join(f"flight{current_flight_index}", "images", img_name)
                    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # 数据处理
                    with data_lock:
                        data_copy = latest_data
                    hex_vals = data_copy.split()
                    row_data = process_hex_data(hex_vals, frame_counter)

                    if csv_writer:
                        csv_writer.writerow(row_data)
                        csv_file.flush()  # 实时保存

                    print(f"\r  [录制] flight{current_flight_index} | 帧:{frame_counter:04d} | Time:{row_data[0]}",
                          end="")

                    frame_counter += 1
                    last_time = current_time

                except Exception as e:
                    print(f"\n✗ 录制错误: {e}")

            prev_recording_state = is_recording
            time.sleep(0.0001)
import serial
import threading
import time
import cv2
import numpy as np
import mss
import os
import json
import re
import csv

# ============== 配置参数 ==============
SERIAL_PORT = 'COM7'
BAUD_RATE = 115200
VIDEO_FPS = 20
CSV_FILENAME = 'telemetry.csv'
CONFIG_FILE = 'config.json'
LOG_FILE = 'check_log.txt'

# 图片目标尺寸 (宽, 高)
TARGET_WIDTH = 128
TARGET_HEIGHT = 80

# ============== 全局变量 ==============
recording_flag = threading.Event()
running = True
latest_data = ""
data_lock = threading.Lock()

# 录制状态资源
current_flight_index = 1
csv_file = None
csv_writer = None
frame_counter = 0

# ============== 启动检查与修复逻辑 (优化版) ==============
def check_and_repair_folders():
    """
    扫描文件夹，仅在发现异常(断层/配置错误)时才写入日志。
    """
    all_items = os.listdir('.')
    flight_pattern = re.compile(r'^flight(\d+)$')
    matched_folders = []

    for item in all_items:
        if os.path.isdir(item):
            match = flight_pattern.match(item)
            if match:
                index = int(match.group(1))
                matched_folders.append((index, item))

    matched_folders.sort(key=lambda x: x[0])

    max_index = 0
    gap_list = []

    if not matched_folders:
        max_index = 0
    else:
        max_index = matched_folders[-1][0]
        expected_index = 1
        for idx, folder_name in matched_folders:
            if idx != expected_index:
                gap_list.append(f"位置 flight{expected_index} 缺失 (当前为 {folder_name})")
            expected_index = idx + 1

    config_needs_update = False
    json_index = 0
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                json_index = config.get('last_index', 0)
        except:
            json_index = 0

    if json_index != max_index:
        config_needs_update = True

    need_repair = len(gap_list) > 0
    need_log = need_repair or config_needs_update

    if not need_log:
        print("✓ 自检正常，文件夹序列连续。")
        return max_index + 1

    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as log_f:
            log_f.write("=" * 40 + "\n")
            log_f.write("启动自检程序 - 发现异常\n")
            log_f.write("=" * 40 + "\n")

            print("⚠ 检测到异常，正在处理... (详情见 check_log.txt)")

            if need_repair:
                log_f.write(f"检测到断层: {gap_list}\n")
                log_f.write("正在执行自动修复 (重排序)...\n")

                new_index = 1
                for old_idx, old_name in matched_folders:
                    new_name = f"flight{new_index}"
                    if old_name != new_name:
                        try:
                            os.rename(old_name, new_name)
                            log_f.write(f"  修复：{old_name} -> {new_name}\n")
                        except Exception as e:
                            log_f.write(f"  错误：无法重命名 {old_name}，{e}\n")
                    new_index += 1

                max_index = new_index - 1
                log_f.write("修复完成。\n")
                print("  -> 文件夹断层已修复")

            if config_needs_update:
                log_f.write(f"配置文件记录({json_index})与实际({max_index})不一致，修正中...\n")
                try:
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump({'last_index': max_index}, f)
                    log_f.write("配置文件已更新。\n")
                    print("  -> 配置文件已同步")
                except Exception as e:
                    log_f.write(f"配置更新失败: {e}\n")

            log_f.write("=" * 40 + "\n")

    except Exception as e:
        print(f"日志写入失败: {e}")

    return max_index + 1

# ============== 录制会话管理 ==============
def start_recording_session(index):
    """开始录制：创建文件夹、打开CSV"""
    global csv_file, csv_writer, frame_counter

    folder_name = f"flight{index}"
    image_folder = os.path.join(folder_name, 'images')

    try:
        os.makedirs(image_folder)
        print(f"▶ [系统] 创建录制目录: {folder_name}/")

        csv_path = os.path.join(folder_name, CSV_FILENAME)
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)

        frame_counter = 0
        print(f"▶ [系统) 数据写入就绪")
    except Exception as e:
        print(f"✗ [错误] 创建失败: {e}")

def stop_recording_session(index):
    """结束录制：关闭CSV、更新配置"""
    global csv_file, csv_writer

    if csv_file:
        csv_file.close()
        print(f"■ [系统] CSV已保存")

    csv_file = None
    csv_writer = None

    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({'last_index': index}, f)
        print(f"■ [系统] 记录已更新: last_index = {index}")
    except Exception as e:
        print(f"✗ [错误] 配置更新失败: {e}")

# ============== 数据处理核心 ==============
def process_hex_data(hex_values, frame_index):
    """
    数据处理逻辑：
    1. 时间 = 帧/20
    2. num = (low+1) + high*256
    3. val = 2*(num/2048) - 1
    4. 交换最后两列
    5. 末列映射 [-1,1] -> [0,1]
    """
    time_seconds = frame_index / 20.0
    calculated_values = []

    for i in range(0, len(hex_values), 2):
        if i + 1 < len(hex_values):
            try:
                byte_low = int(hex_values[i], 16)
                byte_high = int(hex_values[i + 1], 16)

                # 核心公式
                num = (byte_low + 1) + (byte_high * 256)
                value = 2.0 * (num / 2048.0) - 1.0

                calculated_values.append(round(value, 2))
            except:
                calculated_values.append(0.0)

    # 交换最后两列
    if len(calculated_values) >= 2:
        calculated_values[-1], calculated_values[-2] = calculated_values[-2], calculated_values[-1]

    # 末列归一化
    if len(calculated_values) >= 1:
        val_mapped = (calculated_values[-1] + 1) / 2.0
        calculated_values[-1] = round(val_mapped, 2)

    return [str(time_seconds)] + [str(v) for v in calculated_values]

# ============== 线程函数 ==============
def serial_reader():
    """串口监听线程"""
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.001)
        print(f"✓ 串口 {SERIAL_PORT} 已连接")
        buffer = bytearray()
        PACKET_SIZE = 16

        while running:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                buffer.extend(data)

                while len(buffer) >= PACKET_SIZE:
                    packet = buffer[:PACKET_SIZE]
                    buffer = buffer[PACKET_SIZE:]
                    tail = packet[-2:]
                    payload = packet[:8]

                    with data_lock:
                        global latest_data
                        latest_data = ' '.join(f'{b:02X}' for b in payload)

                    # 触发检测
                    if tail == b'\xFF\x07':
                        if not recording_flag.is_set():
                            print(f"\n▶ [触发] 收到开始信号 FF 07")
                            recording_flag.set()
                    elif tail == b'\x00\x00':
                        if recording_flag.is_set():
                            print(f"\n■ [触发] 收到结束信号 00 00")
                            recording_flag.clear()

            time.sleep(0.001)
    except Exception as e:
        print(f"✗ 串口错误: {e}")
    finally:
        if ser: ser.close()

def main_loop():
    """主循环"""
    global current_flight_index, frame_counter, csv_writer

    current_flight_index = check_and_repair_folders()

    print(f"\n系统就绪，图片输出模式: {TARGET_WIDTH}x{TARGET_HEIGHT} 灰度")
    print("等待触发信号...")

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        last_time = time.time()
        interval = 1.0 / VIDEO_FPS
        prev_recording_state = False

        while running:
            current_time = time.time()
            is_recording = recording_flag.is_set()

            # --- 状态机逻辑 ---
            if is_recording and not prev_recording_state:
                start_recording_session(current_flight_index)

            if not is_recording and prev_recording_state:
                stop_recording_session(current_flight_index)
                current_flight_index += 1
                print(f"✓ 就绪。下一录制序号: flight{current_flight_index}")

            # --- 录制核心 ---
            if is_recording and (current_time - last_time >= interval):
                try:
                    # 1. 截图 (原始)
                    screenshot = np.array(sct.grab(monitor))

                    # 2. 转换为灰度图 (黑白)
                    gray_frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)

                    # 3. 强制缩放到 64x40 (忽略原始比例)
                    resized_frame = cv2.resize(gray_frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

                    # 4. 保存图片
                    img_name = f"frame_{frame_counter:06d}.jpg"
                    img_path = os.path.join(f"flight{current_flight_index}", "images", img_name)
                    cv2.imwrite(img_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # 5. 数据处理
                    with data_lock:
                        data_copy = latest_data
                    hex_vals = data_copy.split()
                    row_data = process_hex_data(hex_vals, frame_counter)

                    if csv_writer:
                        csv_writer.writerow(row_data)
                        csv_file.flush()

                    print(f"\r  [录制] flight{current_flight_index} | 帧:{frame_counter:04d} | Time:{row_data[0]}", end="")

                    frame_counter += 1
                    last_time = current_time

                except Exception as e:
                    print(f"\n✗ 录制错误: {e}")

            prev_recording_state = is_recording
            time.sleep(0.0001)

# ============== 入口 ==============
if __name__ == "__main__":
    print("=" * 50)
    print("遥测录制工具 v3.2 (低分辨率灰度版)")
    print(f"图片模式: {TARGET_WIDTH}x{TARGET_HEIGHT} 灰度 (强制比例)")
    print("=" * 50)

    t_serial = threading.Thread(target=serial_reader, daemon=True)
    t_serial.start()

    try:
        main_loop()
    except KeyboardInterrupt:
        running = False
        print("\n程序退出。")



# ============== 入口 ==============
if __name__ == "__main__":
    print("=" * 50)
    print("遥测录制工具 v3.1 (静默启动版)")
    print("功能: 自动修复断层 + 延迟创建 + 实时计算")
    print("=" * 50)

    t_serial = threading.Thread(target=serial_reader, daemon=True)
    t_serial.start()

    try:
        main_loop()
    except KeyboardInterrupt:
        running = False
        print("\n程序退出。")
