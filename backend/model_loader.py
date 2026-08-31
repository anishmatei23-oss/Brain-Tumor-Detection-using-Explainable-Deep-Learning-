import os
from pathlib import Path

import numpy as np
import tensorflow as tf


class BrainTumorModel:
    def __init__(self):
        self.model = None
        self.class_names = ['glioma', 'meningioma', 'pituitary']
        self.img_size = (224, 224)
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            base_dir = Path(__file__).resolve().parent.parent
            possible_paths = [
                base_dir / 'ml_training' / 'models' / 'brain_tumor_cnn.h5',
                base_dir / 'ml_training' / 'models' / 'best_model.h5',
                base_dir / 'backend' / 'models' / 'brain_tumor_cnn.h5',
                Path('models/brain_tumor_cnn.h5'),
                Path('../ml_training/models/brain_tumor_cnn.h5'),
            ]

            for model_path in possible_paths:
                path_str = str(model_path)
                if os.path.exists(path_str):
                    print(f"Loading model from: {path_str}")
                    self.model = tf.keras.models.load_model(path_str)
                    print("Model loaded successfully!")
                    return

            raise FileNotFoundError("Could not find model file in any location")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, img_array):
        """Make prediction on preprocessed image"""
        if self.model is None:
            return self._get_fallback_prediction()
        
        try:
            # Ensure image is properly shaped
            if len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Create detailed predictions for all classes
            detailed_predictions = []
            for i, class_name in enumerate(self.class_names):
                detailed_predictions.append({
                    'type': class_name.capitalize(),
                    'confidence': round(float(predictions[0][i]) * 100, 1)
                })
            
            # Sort by confidence
            detailed_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'success': True,
                'diagnosis': self.class_names[predicted_class_idx].capitalize(),
                'confidence': round(confidence * 100, 1),
                'predictions': detailed_predictions
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._get_fallback_prediction()
    
    def _get_fallback_prediction(self):
        """Return fallback predictions if model fails"""
        return {
            'success': False,
            'diagnosis': 'Unknown',
            'confidence': 0,
            'predictions': [
                {'type': 'Glioma', 'confidence': 33.3},
                {'type': 'Meningioma', 'confidence': 33.3},
                {'type': 'Pituitary', 'confidence': 33.4}
            ]
        }

# Create global instance
brain_tumor_model = BrainTumorModel()