from pathlib import Path
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / 'frontend'
UPLOAD_DIR = BASE_DIR / 'uploads'
HEATMAP_DIR = BASE_DIR / 'static' / 'heatmaps'

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    return send_from_directory(str(FRONTEND_DIR), 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, images)"""
    return send_from_directory(str(FRONTEND_DIR), path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'NeuroVision AI Backend',
        'version': '1.0.0'
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get information about available models"""
    return jsonify({
        'models': [
            {
                'id': 'xception',
                'name': 'Xception',
                'type': 'CNN',
                'accuracy': 94.2,
                'description': 'Extreme version of Inception with depthwise separable convolutions'
            },
            {
                'id': 'resnet50',
                'name': 'ResNet-50',
                'type': 'CNN',
                'accuracy': 95.1,
                'description': 'Deep residual network with skip connections'
            },
            {
                'id': 'inception',
                'name': 'Inception v3',
                'type': 'CNN',
                'accuracy': 92.8,
                'description': 'Multi-scale convolutional blocks'
            },
            {
                'id': 'ensemble',
                'name': 'Ensemble Model',
                'type': 'Ensemble',
                'accuracy': 96.3,
                'description': 'Combined predictions from all models'
            }
        ]
    })

try:
    from backend.routes.upload import upload_bp
    from backend.routes.predict import predict_bp
except ImportError:
    from routes.upload import upload_bp
    from routes.predict import predict_bp

app.register_blueprint(upload_bp)
app.register_blueprint(predict_bp)

if __name__ == '__main__':
    print("Starting NeuroVision AI Backend...")
    print("Serving frontend from:", FRONTEND_DIR)
    print("Server running at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)