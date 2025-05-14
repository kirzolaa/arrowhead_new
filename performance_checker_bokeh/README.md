# Arrowhead Performance Visualizer

A real-time system performance visualization tool using Bokeh for the Arrowhead project.

## Features

- Real-time monitoring of system resources
- Interactive visualization with Bokeh
- Tracks CPU usage, RAM usage, temperatures, and GPU metrics
- Historical data display with time-series charts
- Responsive web interface

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the performance visualizer:

```bash
python performance_visualizer.py
```

This will start a Bokeh server on http://localhost:5006. Open this URL in your web browser to view the real-time performance dashboard.

## Requirements

- Python 3.6+
- Bokeh 3.2.0+
- psutil 5.9.0+
- numpy 1.22.0+
- For GPU monitoring: NVIDIA GPU with nvidia-smi tool installed

## Integration with Arrowhead

This tool can be used to monitor system performance while running Arrowhead components to help identify resource bottlenecks and optimize performance.
