# Model Information

## Model Files

The trained brain tumor classification models are stored in `ml_training/models/`:

- **brain_tumor_cnn.h5** (309 MB) - Main trained CNN model used by the backend
- **best_model.h5** (309 MB) - Duplicate copy
- **brain_tumor_cnn.tflite** (103 MB) - TensorFlow Lite version for mobile/embedded use

## File Size Issue

Due to GitHub's 100MB file size limit, the trained models are not included in this repository.

### Option 1: Using Git LFS (Recommended)
To track large files with Git Large File Storage:

```bash
# Install Git LFS (one-time)
git lfs install

# Track .h5 files
git lfs track "*.h5"
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"

# Then add and push the model
git add ml_training/models/brain_tumor_cnn.h5
git commit -m "Add trained model"
git push
```

### Option 2: Download Pre-trained Model
1. Train your own model using `ml_training/train_first_model.py`
2. Or download from a model registry/release page
3. Place in `ml_training/models/`

### Option 3: Use .gitignore (Current Approach)
Models are excluded from Git. To run the application:
1. Either train a model locally
2. Or download a pre-trained model and place it in `ml_training/models/`

## Model Architecture

The CNN model (`brain_tumor_cnn.h5`) is trained to classify MRI images into:
- **Glioma**
- **Meningioma**
- **Pituitary**

Input size: 224x224x3 RGB images
Output: 3-class classification with confidence scores

## Training Process

To retrain the model:
```bash
cd ml_training
python train_first_model.py
```

The trained model will be saved to `ml_training/models/brain_tumor_cnn.h5`
