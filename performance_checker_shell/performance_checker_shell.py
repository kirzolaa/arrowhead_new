#!/usr/bin/env python3

import psutil
import time
import sys
from datetime import datetime
import platform
import os
import subprocess
import json
from typing import Dict, Optional

class SystemMonitor:
    def __init__(self):
        self.cpu_percent = 0
        self.ram_percent = 0
        self.cpu_temp = 0
        self.gpu_usage = 0
        self.gpu_temp = 0
        self.gpu_vram = 0
        self.last_update = datetime.now()
        
    def get_cpu_info(self):
        return psutil.cpu_percent(interval=0.1)
    
    def get_ram_info(self):
        mem = psutil.virtual_memory()
        return mem.percent
    
    def get_cpu_temp(self) -> float:
        try:
            if platform.system() == "Linux":
                # For Linux systems
                temp = subprocess.check_output(['sensors'])
                temp = temp.decode('utf-8')
                # Extract temperature from output
                for line in temp.split('\n'):
                    if 'Core' in line or 'Package' in line:
                        temp = float(line.split('+')[1].split('°')[0])
                        return temp
            return 0
        except:
            return 0

    def update_gpu_metrics(self):
        try:
            # Get GPU usage
            usage = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
            self.gpu_usage = int(usage.decode().strip())
            
            # Get GPU temperature
            temp = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'])
            self.gpu_temp = int(temp.decode().strip())
            
            # Get GPU VRAM usage
            vram = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
            self.gpu_vram = int(vram.decode().strip())
            
        except Exception as e:
            print(f"Error updating GPU metrics: {e}")
            self.gpu_usage = 0
            self.gpu_temp = 0
            self.gpu_vram = 0
    
    def get_system_info(self) -> Dict[str, str]:
        return {
            'CPU Usage': f"{self.cpu_percent}%",
            'RAM Usage': f"{self.ram_percent}%",
            'CPU Temp': f"{self.cpu_temp}°C",
            'GPU Usage': f"{self.gpu_usage}%",
            'GPU Temp': f"{self.gpu_temp}°C",
            'GPU VRAM': f"{self.gpu_vram} MB",
            'Last Update': self.last_update.strftime('%H:%M:%S')
        }
    
    def update_metrics(self):
        self.cpu_percent = self.get_cpu_info()
        self.ram_percent = self.get_ram_info()
        self.cpu_temp = self.get_cpu_temp()
        self.update_gpu_metrics()
        self.last_update = datetime.now()
    
    def clear_screen(self):
        if platform.system() == "Windows":
            os.system('cls')
        else:
            os.system('clear')
    
    def display_metrics(self):
        self.clear_screen()
        print("\n" + "="*50)
        print(f"System Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
        
        metrics = self.get_system_info()
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("\n" + "="*50)

def main():
    monitor = SystemMonitor()
    try:
        while True:
            monitor.update_metrics()
            monitor.display_metrics()
            time.sleep(1)  # Update every 1 second
    except KeyboardInterrupt:
        print("\nExiting System Monitor...")
        sys.exit(0)

if __name__ == "__main__":
    main()
