#!/usr/bin/env python3

import psutil
import time
import sys
from datetime import datetime
import platform
import os
import subprocess
import json
from typing import Dict, Optional, List
import threading
import numpy as np

# Bokeh imports
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row, grid
from bokeh.models import ColumnDataSource, HoverTool, Range1d, LinearAxis
from bokeh.models.widgets import Div
from bokeh.palettes import Category10
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.themes import Theme

# Maximum number of data points to keep in history
MAX_POINTS = 100

class SystemMonitor:
    def __init__(self):
        self.cpu_percent = 0
        self.ram_percent = 0
        self.cpu_temp = 0
        self.gpu_usage = 0
        self.gpu_temp = 0
        self.gpu_vram = 0
        self.last_update = datetime.now()
        
        # History data
        self.timestamps: List[datetime] = []
        self.cpu_history: List[float] = []
        self.ram_history: List[float] = []
        self.cpu_temp_history: List[float] = []
        self.gpu_usage_history: List[float] = []
        self.gpu_temp_history: List[float] = []
        self.gpu_vram_history: List[float] = []
        
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
        
        # Update history
        self.timestamps.append(self.last_update)
        self.cpu_history.append(self.cpu_percent)
        self.ram_history.append(self.ram_percent)
        self.cpu_temp_history.append(self.cpu_temp)
        self.gpu_usage_history.append(self.gpu_usage)
        self.gpu_temp_history.append(self.gpu_temp)
        self.gpu_vram_history.append(self.gpu_vram)
        
        # Limit history size
        if len(self.timestamps) > MAX_POINTS:
            self.timestamps = self.timestamps[-MAX_POINTS:]
            self.cpu_history = self.cpu_history[-MAX_POINTS:]
            self.ram_history = self.ram_history[-MAX_POINTS:]
            self.cpu_temp_history = self.cpu_temp_history[-MAX_POINTS:]
            self.gpu_usage_history = self.gpu_usage_history[-MAX_POINTS:]
            self.gpu_temp_history = self.gpu_temp_history[-MAX_POINTS:]
            self.gpu_vram_history = self.gpu_vram_history[-MAX_POINTS:]

class BokehVisualizer:
    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor
        self.source = ColumnDataSource(data=dict(
            time=[],
            cpu=[],
            ram=[],
            cpu_temp=[],
            gpu_usage=[],
            gpu_temp=[],
            gpu_vram=[]
        ))
        
        self.update_interval = 1000  # Update interval in milliseconds
        
    def setup_document(self, doc):
        # Create header
        header = Div(text="<h1>Arrowhead System Performance Monitor</h1>", 
                    styles={'text-align': 'center', 'color': '#2F4F4F'})
        
        # Create plots
        # CPU and RAM usage plot
        usage_plot = figure(title="CPU and RAM Usage", 
                           x_axis_label="Time", 
                           y_axis_label="Percentage (%)",
                           height=300, width=800,
                           tools="pan,wheel_zoom,box_zoom,reset,save")
        
        usage_plot.line(x='time', y='cpu', source=self.source, 
                       line_width=2, color=Category10[6][0], 
                       legend_label="CPU Usage")
        
        usage_plot.line(x='time', y='ram', source=self.source, 
                       line_width=2, color=Category10[6][1], 
                       legend_label="RAM Usage")
        
        usage_plot.legend.location = "top_left"
        usage_plot.legend.click_policy = "hide"
        usage_plot.y_range = Range1d(0, 100)
        
        # Temperature plot
        temp_plot = figure(title="Temperature", 
                          x_axis_label="Time", 
                          y_axis_label="Temperature (°C)",
                          height=300, width=800,
                          tools="pan,wheel_zoom,box_zoom,reset,save",
                          x_range=usage_plot.x_range)  # Share x range
        
        temp_plot.line(x='time', y='cpu_temp', source=self.source, 
                      line_width=2, color=Category10[6][2], 
                      legend_label="CPU Temp")
        
        temp_plot.line(x='time', y='gpu_temp', source=self.source, 
                      line_width=2, color=Category10[6][3], 
                      legend_label="GPU Temp")
        
        temp_plot.legend.location = "top_left"
        temp_plot.legend.click_policy = "hide"
        temp_plot.y_range = Range1d(0, 100)
        
        # GPU plot
        gpu_plot = figure(title="GPU Metrics", 
                         x_axis_label="Time", 
                         y_axis_label="Usage (%)",
                         height=300, width=800,
                         tools="pan,wheel_zoom,box_zoom,reset,save",
                         x_range=usage_plot.x_range)  # Share x range
        
        gpu_plot.line(x='time', y='gpu_usage', source=self.source, 
                     line_width=2, color=Category10[6][4], 
                     legend_label="GPU Usage")
        
        # Add secondary y-axis for VRAM
        gpu_plot.extra_y_ranges = {"vram": Range1d(start=0, end=12000)}
        gpu_plot.add_layout(LinearAxis(y_range_name="vram", axis_label="VRAM (MB)"), 'right')
        
        gpu_plot.line(x='time', y='gpu_vram', source=self.source, 
                     line_width=2, color=Category10[6][5], 
                     legend_label="GPU VRAM", y_range_name="vram")
        
        gpu_plot.legend.location = "top_left"
        gpu_plot.legend.click_policy = "hide"
        
        # Add hover tools
        for plot in [usage_plot, temp_plot, gpu_plot]:
            hover = HoverTool()
            hover.tooltips = [
                ("Time", "@time{%H:%M:%S}"),
                ("Value", "@$name")
            ]
            hover.formatters = {"@time": "datetime"}
            plot.add_tools(hover)
        
        # Layout
        layout = column(header, usage_plot, temp_plot, gpu_plot)
        
        # Add to document
        doc.add_root(layout)
        doc.title = "Arrowhead Performance Monitor"
        
        # Add periodic callback
        doc.add_periodic_callback(self.update, self.update_interval)
    
    def update(self):
        # Update monitor data
        self.monitor.update_metrics()
        
        # Update source data
        new_data = {
            'time': self.monitor.timestamps,
            'cpu': self.monitor.cpu_history,
            'ram': self.monitor.ram_history,
            'cpu_temp': self.monitor.cpu_temp_history,
            'gpu_usage': self.monitor.gpu_usage_history,
            'gpu_temp': self.monitor.gpu_temp_history,
            'gpu_vram': self.monitor.gpu_vram_history
        }
        self.source.data = new_data

def bokeh_app(doc):
    monitor = SystemMonitor()
    visualizer = BokehVisualizer(monitor)
    visualizer.setup_document(doc)

def main():
    print("Starting Arrowhead Performance Visualizer...")
    print("Press Ctrl+C to exit")
    
    # Create Bokeh application
    app = Application(FunctionHandler(bokeh_app))
    
    # Create and start Bokeh server
    server = Server({'/': app}, port=5006, allow_websocket_origin=["localhost:5006"])
    server.start()
    
    print("Bokeh server started at http://localhost:5006")
    print("Open your web browser to view the visualization")
    
    try:
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()
    except KeyboardInterrupt:
        print("\nExiting Performance Visualizer...")
        sys.exit(0)

if __name__ == "__main__":
    main()
