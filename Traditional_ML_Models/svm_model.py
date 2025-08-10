import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TurmericSVMDetector:
    def __init__(self, data_path, img_size=(64, 64)):
        self.data_path = data_path
        self.img_size = img_size
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=100)  # Reduce dimensions
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def extract_features(self, image_path):
        """Extract various features from image"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Resize image
            img_resized = cv2.resize(img, self.img_size)
            
            # Feature extraction methods
            features = []
            
            # 1. Color histogram features (RGB)
            for i in range(3):  # RGB channels
                hist = cv2.calcHist([img_resized], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # 2. Gray level features
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            features.extend(gray.flatten())
            
            # 3. HSV color space features
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            for i in range(3):  # HSV channels
                hist = cv2.calcHist([hsv], [i], None, [16], [0, 256])
                features.extend(hist.flatten())
            
            # 4. Edge features (Canny)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # 5. Texture features (Local Binary Pattern approximation)
            # Simple texture measure using standard deviation
            texture_std = np.std(gray)
            texture_mean = np.mean(gray)
            features.extend([texture_std, texture_mean])
            
            # 6. Color moments
            for channel in cv2.split(img_resized):
                mean = np.mean(channel)
                std = np.std(channel)
                skew = np.mean(((channel - mean) / std) ** 3) if std > 0 else 0
                features.extend([mean, std, skew])
                
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_dataset(self):
        """Load and preprocess the dataset"""
        print("🔄 Loading dataset and extracting features...")
        
        X = []
        y = []
        
        # Get class directories
        class_dirs = [d for d in os.listdir(self.data_path) 
                     if os.path.isdir(os.path.join(self.data_path, d))]
        self.class_names = sorted(class_dirs)
        
        print(f"📊 Found {len(self.class_names)} classes: {self.class_names}")
        
        # Process each class
        for class_name in self.class_names:
            class_path = os.path.join(self.data_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"🖼️  Processing {class_name}: {len(image_files)} images")
            
            # Process images with progress bar
            for img_file in tqdm(image_files, desc=f"Extracting {class_name} features"):
                img_path = os.path.join(class_path, img_file)
                features = self.extract_features(img_path)
                
                if features is not None:
                    X.append(features)
                    y.append(class_name)
        
        print(f"✅ Feature extraction completed: {len(X)} samples processed")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def preprocess_data(self, X, y):
        """Preprocess features and labels"""
        print("🔧 Preprocessing data...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        print(f"📐 Original features: {X.shape[1]}, After PCA: {X_pca.shape[1]}")
        print(f"🎯 PCA explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca, y_encoded
    
    def train_model(self, X, y, test_size=0.2):
        """Train SVM model with hyperparameter tuning"""
        print("🚀 Training SVM model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Hyperparameter tuning for SVM
        print("🔍 Performing hyperparameter tuning...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        # Use GridSearchCV for hyperparameter tuning
        svm_model = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(
            svm_model, param_grid, cv=3, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"🎯 Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📈 Test accuracy: {test_accuracy:.4f}")
        
        # Detailed evaluation
        self.evaluate_model(X_test, y_test, y_pred)
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, X_test, y_test, y_pred):
        """Evaluate model performance"""
        print("\n📊 DETAILED EVALUATION RESULTS")
        print("=" * 50)
        
        # Classification report
        class_names_str = [self.class_names[i] for i in range(len(self.class_names))]
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names_str))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names_str, yticklabels=class_names_str)
        plt.title('SVM Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance (for linear kernel)
        if hasattr(self.model, 'coef_') and self.model.kernel == 'linear':
            plt.figure(figsize=(12, 6))
            feature_importance = np.abs(self.model.coef_).mean(axis=0)
            top_indices = np.argsort(feature_importance)[-20:]  # Top 20 features
            
            plt.barh(range(len(top_indices)), feature_importance[top_indices])
            plt.title('Top 20 Most Important Features (Linear SVM)')
            plt.xlabel('Feature Importance (Absolute Coefficient Value)')
            plt.ylabel('Feature Index')
            plt.tight_layout()
            plt.savefig('svm_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def predict_image(self, image_path, confidence_threshold=0.5):
        """Predict disease for a single image"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None, None
        
        # Extract features
        features = self.extract_features(image_path)
        if features is None:
            return None, None
        
        # Preprocess
        features_scaled = self.scaler.transform([features])
        features_pca = self.pca.transform(features_scaled)
        
        # Predict
        prediction = self.model.predict(features_pca)[0]
        probabilities = self.model.predict_proba(features_pca)[0]
        
        # Get class name and confidence
        predicted_class = self.class_names[prediction]
        confidence = np.max(probabilities)
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            return "Low Confidence", confidence
        
        return predicted_class, confidence
    
    def save_model(self, filename='svm_turmeric_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'img_size': self.img_size
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved as {filename}")
    
    def load_model(self, filename='svm_turmeric_model.pkl'):
        """Load a trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.label_encoder = model_data['label_encoder']
            self.class_names = model_data['class_names']
            self.img_size = model_data['img_size']
            
            print(f"✅ Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Main training pipeline"""
    print("🌿 TURMERIC DISEASE DETECTION - SVM MODEL")
    print("=" * 60)
    
    # Initialize detector
    data_path = r"E:\Turmeric dataset\Turmeric Plant Disease\Turmeric Plant Disease"
    detector = TurmericSVMDetector(data_path, img_size=(64, 64))
    
    # Load dataset
    X, y = detector.load_dataset()
    
    if len(X) == 0:
        print("❌ No data loaded. Check your dataset path.")
        return
    
    # Preprocess data
    X_processed, y_processed = detector.preprocess_data(X, y)
    
    # Train model
    X_train, X_test, y_train, y_test = detector.train_model(X_processed, y_processed)
    
    # Save model
    detector.save_model('svm_turmeric_model.pkl')
    
    print("\n🎉 SVM MODEL TRAINING COMPLETED!")
    print("✅ Model saved and ready for use")
    print("📱 Use 'svm_streamlit_app.py' for the web interface")

if __name__ == "__main__":
    main()
