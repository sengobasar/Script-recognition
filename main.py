from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import warnings
import os
import tempfile
import shutil
from typing import Dict, Any
import uvicorn
from datetime import datetime

warnings.filterwarnings("ignore")

# Check for dependencies
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Global model cache for better performance
_trocr_processor = None
_trocr_model = None
_easyocr_reader = None

app = FastAPI(
    title="Cursive Handwriting OCR API",
    description="API for recognizing cursive handwriting from images",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def load_models():
    """Load and cache models for better performance"""
    global _trocr_processor, _trocr_model, _easyocr_reader
    
    if TROCR_AVAILABLE and (_trocr_processor is None or _trocr_model is None):
        print("Loading TrOCR model...")
        _trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        _trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        _trocr_model.eval()
    
    if EASYOCR_AVAILABLE and _easyocr_reader is None:
        print("Loading EasyOCR model...")
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def preprocess_image(image_array):
    """Preprocess image for better OCR results"""
    # Convert to PIL for enhancement
    if len(image_array.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(image_array)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.5)
    
    # Convert back to OpenCV format
    if len(image_array.shape) == 3:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.array(pil_img)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Resize if too small
    height, width = thresh.shape
    if height < 100 or width < 100:
        scale = 2.0
        new_width = int(width * scale)
        new_height = int(height * scale)
        thresh = cv2.resize(thresh, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return thresh

def ocr_with_trocr(image_array):
    """Perform OCR using TrOCR"""
    if not TROCR_AVAILABLE or _trocr_processor is None:
        return None
    
    try:
        # Preprocess image
        processed_img = preprocess_image(image_array)
        
        # Convert to PIL RGB
        pil_image = Image.fromarray(processed_img).convert('RGB')
        
        # Process with TrOCR
        pixel_values = _trocr_processor(images=pil_image, return_tensors="pt").pixel_values
        generated_ids = _trocr_model.generate(pixel_values)
        generated_text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()
    
    except Exception as e:
        print(f"TrOCR error: {e}")
        return None

def ocr_with_easyocr(image_array):
    """Perform OCR using EasyOCR"""
    if not EASYOCR_AVAILABLE or _easyocr_reader is None:
        return None
    
    try:
        # Preprocess image
        processed_img = preprocess_image(image_array)
        
        # Run OCR
        results = _easyocr_reader.readtext(
            processed_img,
            detail=0,
            paragraph=True,
            width_ths=0.4,
            height_ths=0.4
        )
        
        return ' '.join(results) if results else None
    
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return None

async def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file temporarily"""
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{upload_file.filename}"
    file_path = os.path.join("uploads", filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        content = await upload_file.read()
        buffer.write(content)
    
    return file_path

@app.on_event("startup")
async def startup_event():
    """Load models when the server starts"""
    print("Starting Cursive OCR API...")
    load_models()
    print("Models loaded successfully!")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cursive Handwriting OCR</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 2px dashed #ddd;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin-bottom: 20px;
                background-color: #fafafa;
            }
            .upload-area:hover {
                border-color: #007bff;
                background-color: #f0f8ff;
            }
            input[type="file"] {
                display: none;
            }
            .upload-btn {
                background-color: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .upload-btn:hover {
                background-color: #0056b3;
            }
            .process-btn {
                background-color: #28a745;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
                margin-top: 20px;
            }
            .process-btn:hover {
                background-color: #218838;
            }
            .process-btn:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
            }
            .result-area {
                margin-top: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
                display: none;
            }
            .preview-image {
                max-width: 100%;
                max-height: 300px;
                margin: 20px 0;
                border-radius: 5px;
            }
            .loading {
                text-align: center;
                color: #666;
            }
            .error {
                color: #dc3545;
                background-color: #f8d7da;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #f5c6cb;
            }
            .success {
                color: #155724;
                background-color: #d4edda;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #c3e6cb;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🖋️ Cursive Handwriting OCR</h1>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📁 Click here to select an image with cursive handwriting</p>
                <button class="upload-btn" type="button">Choose Image</button>
                <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
            </div>
            
            <div id="imagePreview"></div>
            
            <button class="process-btn" id="processBtn" onclick="processImage()" disabled>
                🔍 Recognize Handwriting
            </button>
            
            <div class="result-area" id="resultArea">
                <h3>Recognition Results:</h3>
                <div id="resultContent"></div>
            </div>
        </div>

        <script>
            let selectedFile = null;

            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    selectedFile = file;
                    
                    // Show preview
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('imagePreview').innerHTML = 
                            `<img src="${e.target.result}" class="preview-image" alt="Preview">`;
                    };
                    reader.readAsDataURL(file);
                    
                    // Enable process button
                    document.getElementById('processBtn').disabled = false;
                }
            }

            async function processImage() {
                if (!selectedFile) {
                    alert('Please select an image first!');
                    return;
                }

                const processBtn = document.getElementById('processBtn');
                const resultArea = document.getElementById('resultArea');
                const resultContent = document.getElementById('resultContent');
                
                // Show loading state
                processBtn.disabled = true;
                processBtn.textContent = '⏳ Processing...';
                resultArea.style.display = 'block';
                resultContent.innerHTML = '<div class="loading">🔄 Analyzing your handwriting...</div>';

                try {
                    const formData = new FormData();
                    formData.append('file', selectedFile);

                    const response = await fetch('/ocr', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        if (result.recognized_text) {
                            resultContent.innerHTML = `
                                <div class="success">
                                    <h4>✅ Text Recognition Successful!</h4>
                                    <p><strong>Method Used:</strong> ${result.method_used}</p>
                                    <p><strong>Processing Time:</strong> ${result.processing_time}s</p>
                                    <p><strong>Recognized Text:</strong></p>
                                    <div style="background: white; padding: 15px; border-radius: 5px; margin: 10px 0; font-family: 'Courier New', monospace; border-left: 4px solid #28a745;">
                                        "${result.recognized_text}"
                                    </div>
                                    <p><strong>Confidence:</strong> ${result.confidence}</p>
                                </div>
                            `;
                        } else {
                            resultContent.innerHTML = `
                                <div class="error">
                                    <h4>❌ No Text Recognized</h4>
                                    <p>Could not detect any text in the image. Try:</p>
                                    <ul>
                                        <li>Ensuring the image is clear and well-lit</li>
                                        <li>Making sure the handwriting is dark on light background</li>
                                        <li>Avoiding shadows or glare</li>
                                        <li>Using a higher resolution image</li>
                                    </ul>
                                </div>
                            `;
                        }
                    } else {
                        resultContent.innerHTML = `
                            <div class="error">
                                <h4>❌ Error Processing Image</h4>
                                <p>${result.detail || 'An error occurred while processing the image.'}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    resultContent.innerHTML = `
                        <div class="error">
                            <h4>❌ Network Error</h4>
                            <p>Failed to connect to the server. Please try again.</p>
                        </div>
                    `;
                }

                // Reset button
                processBtn.disabled = false;
                processBtn.textContent = '🔍 Recognize Handwriting';
            }

            // Drag and drop functionality
            const uploadArea = document.querySelector('.upload-area');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#007bff';
                uploadArea.style.backgroundColor = '#f0f8ff';
            });
            
            uploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#ddd';
                uploadArea.style.backgroundColor = '#fafafa';
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#ddd';
                uploadArea.style.backgroundColor = '#fafafa';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    document.getElementById('fileInput').files = files;
                    handleFileSelect({target: {files: files}});
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Perform OCR on uploaded image
    
    Returns:
        Dictionary containing recognition results
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    
    # Check file size (10MB limit)
    file_size = 0
    temp_file_path = None
    
    try:
        # Save uploaded file
        temp_file_path = await save_uploaded_file(file)
        file_size = os.path.getsize(temp_file_path)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File too large. Please upload an image smaller than 10MB")
        
        # Read image
        image_array = cv2.imread(temp_file_path)
        if image_array is None:
            raise HTTPException(status_code=400, detail="Could not read the image. Please upload a valid image file")
        
        # Start timing
        start_time = datetime.now()
        
        # Try TrOCR first
        result = ocr_with_trocr(image_array)
        method_used = "TrOCR"
        confidence = "High" if result and len(result.strip()) > 3 else "Low"
        
        # If TrOCR fails, try EasyOCR
        if not result or len(result.strip()) < 3:
            result = ocr_with_easyocr(image_array)
            method_used = "EasyOCR"
            confidence = "Medium" if result and len(result.strip()) > 3 else "Low"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response_data = {
            "success": True,
            "recognized_text": result if result else None,
            "method_used": method_used,
            "processing_time": round(processing_time, 2),
            "confidence": confidence,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "trocr_available": TROCR_AVAILABLE,
        "easyocr_available": EASYOCR_AVAILABLE,
        "models_loaded": {
            "trocr": _trocr_processor is not None,
            "easyocr": _easyocr_reader is not None
        }
    }

@app.get("/models/info")
async def models_info():
    """Get information about available models"""
    return {
        "available_models": {
            "trocr": {
                "available": TROCR_AVAILABLE,
                "loaded": _trocr_processor is not None,
                "description": "Microsoft TrOCR - Best for handwritten text"
            },
            "easyocr": {
                "available": EASYOCR_AVAILABLE,
                "loaded": _easyocr_reader is not None,
                "description": "EasyOCR - General purpose OCR"
            }
        },
        "recommendation": "TrOCR for best cursive handwriting recognition"
    }

if __name__ == "__main__":
    print("🚀 Starting Cursive OCR FastAPI Server...")
    print("📝 Open http://localhost:8000 in your browser")
    print("🔧 API docs available at http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",  # Change this to match your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )