🖋️ CursiveAI - Advanced Handwriting Recognition API
Show Image
Show Image
Show Image
Show Image

A powerful, AI-driven web application for recognizing cursive and handwritten text from images. Built with FastAPI and powered by Microsoft's TrOCR and EasyOCR.

✨ Features

🎯 Specialized Cursive Recognition - Optimized for cursive and handwritten text
🚀 Dual AI Engines - TrOCR (transformer-based) + EasyOCR fallback
🌐 Beautiful Web Interface - Drag & drop image upload with real-time processing
⚡ Fast Processing - Model caching for quick subsequent recognitions
📱 Responsive Design - Works perfectly on desktop and mobile
🔧 RESTful API - Easy integration with other applications
📊 Detailed Results - Processing time, confidence scores, and method used
🛡️ Error Handling - Comprehensive validation and user-friendly error messages

🚀 Quick Start
Prerequisites

Python 3.8 or higher
pip package manager

Installation

Clone the repository
bashgit clone https://github.com/yourusername/cursive-ai.git
cd cursive-ai

Create a virtual environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
bashpip install -r requirements.txt

Run the application
bashpython main.py

Open your browser
http://localhost:8000


That's it! 🎉
📸 Screenshots
Web Interface
Show Image
Results Display
Show Image
🛠️ Technology Stack

Backend: FastAPI (Python)
AI Models:

Microsoft TrOCR (Transformer-based OCR)
EasyOCR (Traditional OCR)


Image Processing: OpenCV, PIL
Frontend: HTML5, CSS3, JavaScript (embedded)
Deep Learning: PyTorch, Transformers

📚 API Documentation
Endpoints
MethodEndpointDescriptionGET/Web interfacePOST/ocrUpload image for OCR processingGET/healthHealth check and model statusGET/models/infoInformation about available modelsGET/docsInteractive API documentation
Example API Usage
bash# Upload an image for OCR processing
curl -X POST "http://localhost:8000/ocr" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_handwriting.jpg"
Response:
json{
  "success": true,
  "recognized_text": "Your handwritten text here",
  "method_used": "TrOCR",
  "processing_time": 2.34,
  "confidence": "High",
  "file_size_mb": 0.85,
  "timestamp": "2024-01-15T10:30:45"
}
🎯 Use Cases

Document Digitization - Convert handwritten notes to digital text
Historical Document Processing - Digitize old letters and manuscripts
Educational Tools - Help students digitize their handwritten work
Business Applications - Process handwritten forms and surveys
Research Projects - Analyze handwritten data at scale

📁 Project Structure
cursive-ai/
│
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # MIT License
│
├── uploads/               # Temporary file storage (auto-created)
├── static/                # Static files (auto-created)
│
└── tests/                 # Test files (optional)
    └── test_main.py
🔧 Configuration
Environment Variables
Create a .env file for custom configuration:
env# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Model Configuration
USE_GPU=False
MAX_FILE_SIZE_MB=10
CACHE_MODELS=True
Advanced Settings
Modify these settings in main.py:
python# Model cache settings
ENABLE_MODEL_CACHING = True

# File upload limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# OCR processing options
PREFER_TROCR = True  # Set to False to prefer EasyOCR for speed
🚀 Deployment
Local Development
bashpython main.py
Production (with Gunicorn)
bashpip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
Docker Deployment
dockerfileFROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
Cloud Deployment
Deploy easily to:

Heroku: git push heroku main
Railway: Connect GitHub repo
Vercel: Deploy with FastAPI preset
DigitalOcean: Use App Platform

🎨 Customization
Modify the Web Interface
The HTML interface is embedded in main.py. Search for the read_root() function to customize:

Styling: Modify the CSS in the <style> section
Functionality: Update JavaScript for new features
Layout: Change HTML structure for different UI

Add New OCR Models
python# Add your custom OCR model
def ocr_with_custom_model(image_array):
    # Your implementation here
    return recognized_text
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
Development Setup

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

Running Tests
bashpip install pytest pytest-asyncio
pytest tests/
📊 Performance

First Recognition: 3-5 seconds (model loading)
Subsequent Recognitions: 1-3 seconds
Memory Usage: ~2GB (with models loaded)
Supported Image Formats: JPG, PNG, BMP, TIFF
Max Image Size: 10MB (configurable)

🔍 Troubleshooting
Common Issues
1. Models not loading
bash# Clear model cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.EasyOCR/
2. Out of memory errors
python# Disable GPU in main.py
gpu = False  # Change this line
3. Slow processing

Use GPU if available
Reduce image size before upload
Enable model caching

Getting Help

📖 Check the Wiki
🐛 Report bugs in Issues
💬 Join discussions in Discussions

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Microsoft TrOCR Team - For the amazing transformer-based OCR model
EasyOCR Contributors - For the robust traditional OCR solution
FastAPI Community - For the excellent web framework
Hugging Face - For model hosting and transformers library

🌟 Star History
Show Image

Made with ❤️ for the OCR community
