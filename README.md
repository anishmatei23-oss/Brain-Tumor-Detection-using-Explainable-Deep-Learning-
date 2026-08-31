# NeuroVision AI: Brain Tumor Detection using Explainable Deep Learning

A full-stack brain tumor classification project that combines a Flask backend, a responsive frontend, and a trained deep learning model for Glioma, Meningioma, and Pituitary tumor detection.

## Features

- MRI upload and preprocessing
- Brain tumor classification using a trained Keras model
- Explainable AI heatmap output via Grad-CAM style visualization
- Responsive web interface for upload and diagnosis
- Health and model API endpoints
- Demo prediction mode for sample testing

## Project Structure

- `backend/` – Flask API and model logic
- `frontend/` – HTML, CSS, and JavaScript UI
- `ml_training/` – training scripts and saved model files
- `uploads/` – uploaded MRI files
- `static/heatmaps/` – generated heatmap outputs

## Tech Stack

- Python 3.10+
- Flask
- TensorFlow / Keras
- OpenCV
- Pillow
- pydicom
- JavaScript / HTML / CSS

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate it:
   - Windows PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Windows CMD:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the backend:
   ```bash
   python backend/app.py
   ```
6. Open the browser to:
   ```text
   http://localhost:5000
   ```

## Model Notes

The project includes a trained model at:

- `ml_training/models/brain_tumor_cnn.h5`

If the model is missing or replaced, update the loading paths in [backend/model_loader.py](backend/model_loader.py).

## API Endpoints

- `GET /api/health`
- `POST /api/upload`
- `POST /api/analyze`
- `POST /api/predict/demo`
- `GET /api/predict/history`
- `GET /api/predict/stats`

## Verification

The project was validated end-to-end with a generated MRI-like test image and returned successful upload and diagnosis responses:

- Upload status: `200`
- Analysis status: `200`
- Diagnosis: `Pituitary`
- Confidence: `62.9%`

## License

This project is intended for educational and research use.
