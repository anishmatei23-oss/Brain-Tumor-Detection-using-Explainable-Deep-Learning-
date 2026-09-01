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
- `brain_tumor_training.py` – downloads the dataset and trains the classifier
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
5. Train a model. The public Kaggle dataset is downloaded to KaggleHub's local cache:
   ```bash
   python brain_tumor_training.py
   ```
   Configure Kaggle authentication as required by KaggleHub before running this command.
6. Start the backend:
   ```bash
   python backend/app.py
   ```
7. Open the browser to:
   ```text
   http://localhost:5000
   ```

## Model Notes

The trained model is generated locally at `backend/models/brain_tumor_cnn.keras`.
Model files are ignored by Git because GitHub rejects files larger than 100 MB.

The trainer uses `masoudnickparvar/brain-tumor-mri-dataset`, ignores its
`notumor` class to match the backend's three-class model, and creates the
validation split without copying images into the repository.

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
