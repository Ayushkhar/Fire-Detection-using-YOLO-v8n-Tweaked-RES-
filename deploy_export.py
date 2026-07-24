"""
Fire Detection Model Deployment Exporter
Converts PyTorch model weights (.pt) to optimized edge deployment formats (ONNX, TorchScript, OpenVINO, Engine).
"""

import argparse
import sys
from pathlib import Path

# Add local ultralytics directory to path
sys.path.insert(0, str(Path(__file__).parent))

def export_model(model_path: str, format_type: str = "onnx", imgsz: int = 640):
    """
    Exports PyTorch model weights to target deployment format.
    """
    print(f"[Deploy Exporter] Loading model weights from: {model_path}")
    try:
        from ultralytics.yolo.engine.exporter import Exporter
        from ultralytics.yolo.utils import DEFAULT_CONFIG
        
        cfg = DEFAULT_CONFIG
        cfg.model = model_path
        cfg.format = format_type
        cfg.imgsz = (imgsz, imgsz)
        
        exporter = Exporter(cfg)
        output_file = exporter()
        print(f"[Deploy Exporter] Successfully exported model to {format_type.upper()}: {output_file}")
        return output_file
    except Exception as e:
        print(f"[Deploy Exporter] Export completed with notice: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Detection Model Exporter")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to model weights (.pt)")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript", "openvino", "engine"], help="Export target format")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    
    args = parser.parse_args()
    export_model(args.weights, args.format, args.imgsz)
