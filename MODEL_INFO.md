# Model Information

`brain_tumor_training.py` downloads the public Kaggle brain tumor MRI dataset
and trains a three-class MobileNetV2 classifier for Glioma, Meningioma, and
Pituitary tumors.

The generated model is saved to `backend/models/brain_tumor_cnn.keras`. Model
files, datasets, logs, uploads, and virtual environments are excluded from
Git. Run the trainer whenever a local model is needed:

```bash
python brain_tumor_training.py
```

This project is intended for educational and research use, not clinical
diagnosis.
