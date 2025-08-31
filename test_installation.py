print("🧪 Testing OCR Installation...")

try:
    import fastapi
    print("✅ FastAPI: OK")
except ImportError as e:
    print(f"❌ FastAPI: {e}")

try:
    import cv2
    print("✅ OpenCV: OK")
except ImportError as e:
    print(f"❌ OpenCV: {e}")

try:
    import numpy
    print("✅ NumPy: OK")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    from PIL import Image
    print("✅ PIL/Pillow: OK")
except ImportError as e:
    print(f"❌ PIL/Pillow: {e}")

try:
    import easyocr
    print("✅ EasyOCR: OK")
except ImportError as e:
    print(f"❌ EasyOCR: {e}")

try:
    from transformers import TrOCRProcessor
    print("✅ Transformers (TrOCR): OK")
except ImportError as e:
    print(f"❌ Transformers: {e}")

try:
    import torch
    print("✅ PyTorch: OK")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

print("\n🚀 Installation test complete!")