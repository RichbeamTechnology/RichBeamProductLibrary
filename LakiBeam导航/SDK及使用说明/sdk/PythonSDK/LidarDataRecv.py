import threading
import socket
import numpy as np
import time
import requests
import tkinter as tk
from tkinter import scrolledtext, messagebox


ANGLE_RESOLUTION = {
    10: 0.1,
    15: 0.25,
    20: 0.25,
    25: 0.25,
    30: 0.25,
}


class LidarReceiver:
    def __init__(self, root, lidar_ip, host_ip, udp_port):
        self.root = root
        self.root.title("Lidar Data Output")

        self.text_area = scrolledtext.ScrolledText(root, width=80, height=30)
        self.text_area.pack(padx=10, pady=10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 10)
        self.sock.bind((host_ip, udp_port))

        self.data_buffer = {}
        self.running = True          # 程序是否正在运行（用于线程退出）
        self.paused = False           # 是否处于暂停状态
        self.lock = threading.Lock()
        self.last_timestamp = 0

        # 获取雷达频率
        try:
            url = f"http://{lidar_ip}/api/v1/sensor/scanfreq"
            lidar_frequency = int(requests.get(url, timeout=2).json()["scanfreq"]) 
        except Exception:
            lidar_frequency = 10
            self.log("无法读取雷达频率，默认使用 10Hz")

        self.lidar_frequency = lidar_frequency

        # 绑定空格键：暂停/继续接收
        self.root.bind_all("<space>", self.toggle_pause)

        self.start_receiving()

    def log(self, message):
        self.text_area.insert(tk.END, message + "\n")
        self.text_area.yview(tk.END)

    def start_receiving(self):
        thread = threading.Thread(target=self.receive_data, daemon=True)
        thread.start()

    def toggle_pause(self, event=None):
        """空格键：暂停或恢复接收"""
        self.paused = not self.paused
        if self.paused:
            self.log("已暂停接收数据（按空格键可继续）")
        else:
            self.log("已恢复接收数据")

    def receive_data(self):
        """循环接收数据（可被暂停）"""
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                data, _ = self.sock.recvfrom(1206)
                self.process_packet(data)
            except Exception as e:
                self.log(f"Receiver error: {e}")
                time.sleep(0.1)

    def process_packet(self, data):
        packet = memoryview(data)
        if len(packet) < 4 or packet[0] != 0xFF or packet[1] != 0xEE:
            return

        if len(packet) >= 1206:
            self.last_timestamp = int.from_bytes(packet[1200:1204], 'little')

        angle_step = ANGLE_RESOLUTION.get(self.lidar_frequency, 0.25)
        block_angle_range = 1.6 if self.lidar_frequency == 10 else 4.0
        angle_increments = np.linspace(0, block_angle_range,
                                       num=int(block_angle_range / angle_step),
                                       endpoint=False)

        with self.lock:
            for block_idx in range(12):
                block_start = block_idx * 100
                if block_start + 100 > len(packet):
                    break
                if packet[block_start] != 0xFF or packet[block_start + 1] != 0xEE:
                    continue

                azimuth = int.from_bytes(packet[block_start + 2:block_start + 4], 'little') / 100.0

                for i in range(len(angle_increments)):
                    offset = block_start + 4 + i * 6
                    distance = int.from_bytes(packet[offset:offset + 2], 'little')
                    rssi = packet[offset + 2]
                    if distance == 0xFFFF:
                        continue
                    angle = (azimuth + angle_increments[i]) % 360
                    self.data_buffer[angle] = (distance / 1000.0, rssi)

            if len(self.data_buffer) >= int(360 / angle_step):
                self.log(f"--- 新的一圈数据 ---")
                self.log(f"Frequency: {self.lidar_frequency}Hz, AngleRes: {angle_step}°, Points: {len(self.data_buffer)}")
                self.log(f"Last packet timestamp: {self.last_timestamp} (raw units)")
                self.output_data()
                self.data_buffer.clear()

    def output_data(self):
        output_lines = [
            f"Angle={angle:.2f}°，Distance={dist:.3f}m，RSSI={rssi}"
            for angle, (dist, rssi) in sorted(self.data_buffer.items(), key=lambda x: x[0])
        ]
        self.log("\n".join(output_lines))


# ===========================
# 连接配置界面
# ===========================

class ConnectionWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Lidar Connection Setup")

        tk.Label(root, text="主机 IP:").grid(row=0, column=0, padx=5, pady=5)
        tk.Label(root, text="雷达 IP:").grid(row=1, column=0, padx=5, pady=5)
        tk.Label(root, text="端口号:").grid(row=2, column=0, padx=5, pady=5)

        self.host_ip_entry = tk.Entry(root)
        self.lidar_ip_entry = tk.Entry(root)
        self.port_entry = tk.Entry(root)

        self.host_ip_entry.grid(row=0, column=1, padx=5, pady=5)
        self.lidar_ip_entry.grid(row=1, column=1, padx=5, pady=5)
        self.port_entry.grid(row=2, column=1, padx=5, pady=5)

        self.host_ip_entry.insert(0, "192.168.8.1")
        self.lidar_ip_entry.insert(0, "192.168.8.2")
        self.port_entry.insert(0, "2368")

        connect_button = tk.Button(root, text="连接雷达", command=self.connect)
        connect_button.grid(row=3, column=0, columnspan=2, pady=10)

        # 提示信息：说明空格键的作用
        tip_label = tk.Label(
            root,
            text="提示：连接后可通过【空格键】暂停/继续接收数据。",
            fg="gray"
        )
        tip_label.grid(row=4, column=0, columnspan=2, pady=5)

    def connect(self):
        host_ip = self.host_ip_entry.get().strip()
        lidar_ip = self.lidar_ip_entry.get().strip()
        port_text = self.port_entry.get().strip()

        if not (host_ip and lidar_ip and port_text.isdigit()):
            messagebox.showerror("错误", "请输入正确的 IP 和端口号！")
            return

        udp_port = int(port_text)

        self.root.destroy()
        main_root = tk.Tk()
        LidarReceiver(main_root, lidar_ip, host_ip, udp_port)
        main_root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    ConnectionWindow(root) 
    root.mainloop()
