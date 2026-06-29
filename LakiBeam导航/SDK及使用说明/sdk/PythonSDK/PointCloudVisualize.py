import PyQt5
import threading
import queue
import time
import requests
import socket
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.use('Qt5Agg')

# 配置参数
LIDAR_IP = "192.168.101.178"
HOST_IP = "192.168.101.115"
UDP_PORT = 2369
BUFFER_SIZE = 1024 * 1024 * 20  # 20MB UDP缓冲区

r = requests.get(f"http://{LIDAR_IP}/api/v1/sensor/scanfreq")
scan_frequency = json.loads(r.text)["scanfreq"] # 读取当前雷达频率
print(scan_frequency)

if scan_frequency == 10:
        MAX_POINTS = 3600
else:
        MAX_POINTS = 1440
#MAX_POINTS # 最大显示点数

class LidarProcessor:
    def __init__(self):

        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.scatter = self.ax.scatter([], [], s=1, c='blue', alpha=0.6)
        self.ax.set_title('Real-time LiDAR (0-360°)')
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='datalim')
        
        self.zoom_scale = 1.2  # 缩放比例
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.data_queue = queue.Queue(maxsize=20)  # 限制队列大小防止堆积
        self.current_points = np.zeros((MAX_POINTS, 2)) * np.nan
        self.point_index = 0
        self.last_azimuth = None
        self.lock = threading.Lock()
    
    def start_animation(self):
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, cache_frame_data=False)
        plt.show()
    

    def update_plot(self, frame):
        """非阻塞式更新绘图"""
        try:
            while True:
                # 批量获取队列中的数据
                new_points = self.data_queue.get_nowait()
                with self.lock:
                    for x, y in new_points:
                        self.current_points[self.point_index % MAX_POINTS] = [x, y]
                        self.point_index += 1
        except queue.Empty:
            pass
        
        # 过滤无效数据
        valid_points = self.current_points[~np.isnan(self.current_points).any(axis=1)]
        
        # 更新散点图数据
        self.scatter.set_offsets(valid_points[-MAX_POINTS:])  # 仅保留最新数据
        return self.scatter,

    def on_scroll(self, event):
        """滚轮事件：放大或缩小坐标轴"""
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata  # 鼠标位置
        ydata = event.ydata

        if xdata is None or ydata is None:
            return  # 鼠标不在绘图区内

        scale_factor = self.zoom_scale if event.button == 'down' else 1 / self.zoom_scale

        new_xlim = [
            xdata - (xdata - cur_xlim[0]) * scale_factor,
            xdata + (cur_xlim[1] - xdata) * scale_factor
        ]
        new_ylim = [
            ydata - (ydata - cur_ylim[0]) * scale_factor,
            ydata + (cur_ylim[1] - ydata) * scale_factor
        ]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()

def data_receiver(processor):
    """独立数据接收线程"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
    sock.bind((HOST_IP, UDP_PORT))
    
    batch_points = []
    last_flush_time = time.time()
    
    while True:
        try:
            data, _ = sock.recvfrom(1206)
            points = parse_packet(data)
            
            if points:
                batch_points.extend(points)
                
                # 批量提交（每50ms或满100点提交一次）
                if (time.time() - last_flush_time > 0.005) or (len(batch_points) >= 50):
                    try:
                        processor.data_queue.put_nowait(batch_points)
                        batch_points = []
                        last_flush_time = time.time()
                    except queue.Full:
                        batch_points = []  # 丢弃旧数据保实时性
                        
        except Exception as e:
            print(f"Receiver error: {e}")

def parse_packet(data):

    packet = bytearray(data)
    if len(packet) < 4 or packet[0] != 0xFF or packet[1] != 0xEE:
        return []
    
    points = []
    
    # 预计算角度增量(10Hz:0.1°; 20/25/30Hz: 0.25°)
    if scan_frequency == 10:
        angle_step = 1.6 / 16
    else:
        angle_step = 4.0 / 16

    angle_increments = np.arange(16) * angle_step

    
    for block_idx in range(12):
        block_start = block_idx * 100
        if block_start + 100 > len(packet):
            break
            
        # 快速校验包头
        if packet[block_start] != 0xFF or packet[block_start+1] != 0xEE:
            continue
            
        # 批量读取方位角
        azimuth = int.from_bytes(packet[block_start+2:block_start+4], 'little') / 100.0
        
        # 处理16个点
        for i in range(16):
            offset = block_start + 4 + i*6
            distance = int.from_bytes(packet[offset:offset+2], 'little')
            rssi = packet[offset+2]

            if distance == 0xFFFF:
                continue
                
            # 向量化计算
            angle = azimuth + angle_increments[i]
            angle_vec = np.deg2rad(azimuth + angle_increments[i])
            dist_m = distance / 1000.0
            x = dist_m * np.cos(angle_vec)
            y = dist_m * np.sin(angle_vec)
            
            points.append((x, y))


    return points

if __name__ == "__main__":
    # 设置扫描频率
    
    
    # 启动处理系统
    processor = LidarProcessor()
    
    # 启动独立接收线程
    receiver_thread = threading.Thread(
        target=data_receiver, 
        args=(processor,), 
        daemon=True
    )
    receiver_thread.start()
    
    # 启动可视化
    processor.start_animation()


