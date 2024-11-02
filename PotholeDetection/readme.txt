Step 1 | Install ultrolytics library in terminal or powershell.
pip3 install ultralytics

Step 2 | Check Hardware requirements for YOLOv8's models.
| Model    | Parameters (M) | GFLOPs | Recommended Hardware                           |
|----------|----------------|--------|------------------------------------------------|
| YOLOv8n  | 3.2            | 8.7    | CPU: 4GB RAM, Jetson Nano: 4GB VRAM            |
| YOLOv8s  | 11.2           | 28.6   | Jetson Orin: 8GB RAM, Low-power GPUs: 4GB VRAM |
| YOLOv8m  | 25.9           | 78.9   | Desktop GPU: 8GB VRAM, 8GB+ RAM                |
| YOLOv8l  | 43.7           | 157.8  | High-end GPU: 12GB VRAM, 16GB+ RAM             |
| YOLOv8x  | 68.2           | 257.8  | Powerful GPU: 24GB VRAM, 32GB+ RAM             |

Step 3 | Loading YOLOv8-seg Pre-Trained Model
| Model       | Size (pixels) | mAP^box | mAP^mask | Speed    | Speed    | params | FLOPs |
|             |               | 50-95   | 50-95    | CPU ONNX | A100     | (M)    | (8)   |
|             |               |         |          | (ms)     | TensorRT |        |       |
|             |               |         |          |          | (ms)     |        |       |
|-------------|---------------|---------|----------|----------|----------|--------|-------|
| YOLOv8n-seg | 640           | 36.7    | 30.5     | 96.1     | 1.21     | 3.4    | 12.6  |
| YOLOv8s-seg | 640           | 44.6    | 44.6     | 155.7    | 1.47     | 11.8   | 42.6  |
| YOLOv8m-seg | 640           | 49.9    | 49.9     | 317.0    | 2.18     | 27.3   | 110.2 |
| YOLOv8l-seg | 640           | 52.3    | 52.3     | 572.4    | 2.79     | 46.0   | 220.5 |
| YOLOv8x-seg | 640           | 53.4    | 53.4     | 712.1    | 4.02     | 71.8   | 344.1 |

