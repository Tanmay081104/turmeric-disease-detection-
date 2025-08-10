import os
import numpy as np
import cv2
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TurmericNaiveBayesDetector:
    def __init__(self, data_path, img_size=(64, 64), nb_variant='gaussian'):
        self.data_path = data_path
        self.img_size = img_size
        self.nb_variant = nb_variant
        self.model = None
        self.scaler = StandardScaler() if nb_variant == 'gaussian' else MinMaxScaler()
        self.feature_selector = SelectKBest(f_classif, k=50)  # Select best features
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def extract_features(self, image_path):
        """Extract comprehensive features from image"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Resize image
            img_resized = cv2.resize(img, self.img_size)
            
            features = []
            
            # 1. Color histogram features (RGB) - more bins for NB
            for i in range(3):  # RGB channels
                hist = cv2.calcHist([img_resized], [i], None, [16], [0, 256])
                features.extend(hist.flatten())
            
            # 2. HSV color histogram
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            for i in range(3):  # HSV channels
                hist = cv2.calcHist([hsv], [i], None, [12], [0, 256])
                features.extend(hist.flatten())
            
            # 3. Gray level co-occurrence matrix features (simplified)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Statistical features from gray image
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.var(gray),
                np.min(gray),
                np.max(gray),
                np.median(gray),
                np.percentile(gray, 25),
                np.percentile(gray, 75)
            ])
            
            # 4. Edge density features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # 5. Texture features using local variance
            kernel = np.ones((3,3), np.float32) / 9
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
            features.extend([
                np.mean(local_variance),
                np.std(local_variance),
                np.max(local_variance),
                np.min(local_variance)
            ])
            
            # 6. Color moments for each channel
            for channel in cv2.split(img_resized):
                mean = np.mean(channel)
                std = np.std(channel)
                skew = np.mean(((channel - mean) / std) ** 3) if std > 0 else 0
                kurtosis = np.mean(((channel - mean) / std) ** 4) if std > 0 else 0
                features.extend([mean, std, skew, kurtosis])
                
            # 7. Shape features (using contours)
            edges_for_contours = cv2.Canny(gray, 100, 200)
            contours, _ = cv2.findContours(edges_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Shape features
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    features.append(circularity)
                else:
                    features.append(0)
                    
                features.extend([area, perimeter])
            else:
                features.extend([0, 0, 0])  # No contours found
            
            # 8. Green color dominance (important for leaf diseases)
            hsv_mean = np.mean(hsv, axis=(0,1))
            green_dominance = hsv_mean[1]  # Saturation in green range
            features.append(green_dominance)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_dataset(self):
        """Load and preprocess the dataset"""
        print("🔄 Loading dataset and extracting features for Naive Bayes...")
        
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
        """Preprocess features and labels for Naive Bayes"""
        print("🔧 Preprocessing data for Naive Bayes...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Handle NaN/infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Scale features based on NB variant
        if self.nb_variant == 'gaussian':
            X_scaled = self.scaler.fit_transform(X)
        else:
            # For Multinomial/Complement NB, ensure non-negative features
            X_scaled = self.scaler.fit_transform(np.abs(X))
        
        # Feature selection
        X_selected = self.feature_selector.fit_transform(X_scaled, y_encoded)
        
        print(f"📐 Original features: {X.shape[1]}, After selection: {X_selected.shape[1]}")
        
        return X_selected, y_encoded
    
    def train_model(self, X, y, test_size=0.2):
        """Train Naive Bayes model"""
        print(f"🚀 Training Naive Bayes model ({self.nb_variant})...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Choose Naive Bayes variant
        if self.nb_variant == 'gaussian':
            self.model = GaussianNB()
        elif self.nb_variant == 'multinomial':
            self.model = MultinomialNB()
        else:  # complement
            self.model = ComplementNB()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"🎯 Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Test set evaluation
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=class_names_str, yticklabels=class_names_str)
        plt.title(f'{self.nb_variant.capitalize()} Naive Bayes - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('nb_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance (for selected features)
        if hasattr(self.feature_selector, 'scores_'):
            plt.figure(figsize=(12, 6))
            selected_indices = self.feature_selector.get_support(indices=True)
            feature_scores = self.feature_selector.scores_[selected_indices]
            
            # Plot top features
            top_indices = np.argsort(feature_scores)[-20:]  # Top 20 features
            
            plt.barh(range(len(top_indices)), feature_scores[top_indices])
            plt.title('Top 20 Most Important Features (F-score)')
            plt.xlabel('F-score')
            plt.ylabel('Feature Rank')
            plt.tight_layout()
            plt.savefig('nb_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def predict_image(self, image_path, confidence_threshold=0.4):
        """Predict disease for a single image"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None, None
        
        # Extract features
        features = self.extract_features(image_path)
        if features is None:
            return None, None
        
        # Handle NaN/infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Preprocess
        if self.nb_variant == 'gaussian':
            features_scaled = self.scaler.transform([features])
        else:
            features_scaled = self.scaler.transform([np.abs(features)])
            
        features_selected = self.feature_selector.transform(features_scaled)
        
        # Predict
        prediction = self.model.predict(features_selected)[0]
        probabilities = self.model.predict_proba(features_selected)[0]
        
        # Get class name and confidence
        predicted_class = self.class_names[prediction]
        confidence = np.max(probabilities)
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            return "Low Confidence", confidence
        
        return predicted_class, confidence
    
    def save_model(self, filename=None):
        """Save the trained model"""
        if filename is None:
            filename = f'{self.nb_variant}_nb_turmeric_model.pkl'
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'img_size': self.img_size,
            'nb_variant': self.nb_variant
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved as {filename}")
    
    def load_model(self, filename):
        """Load a trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.label_encoder = model_data['label_encoder']
            self.class_names = model_data['class_names']
            self.img_size = model_data['img_size']
            self.nb_variant = model_data['nb_variant']
            
            print(f"✅ Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def compare_nb_variants():
    """Compare different Naive Bayes variants"""
    print("🔬 COMPARING NAIVE BAYES VARIANTS")
    print("=" * 50)
    
    data_path = r"E:\Turmeric dataset\Turmeric Plant Disease\Turmeric Plant Disease"
    variants = ['gaussian', 'multinomial', 'complement']
    results = {}
    
    for variant in variants:
        print(f"\n🧪 Testing {variant.upper()} Naive Bayes...")
        
        # Initialize detector
        detector = TurmericNaiveBayesDetector(data_path, nb_variant=variant)
        
        # Load dataset (only once, reuse for all variants)
        if 'X' not in locals():
            X, y = detector.load_dataset()
        
        # Preprocess data
        X_processed, y_processed = detector.preprocess_data(X, y)
        
        # Train model
        X_train, X_test, y_train, y_test = detector.train_model(X_processed, y_processed)
        
        # Store results
        y_pred = detector.model.predict(X_test)
        results[variant] = accuracy_score(y_test, y_pred)
        
        # Save model
        detector.save_model()
    
    # Display comparison
    print("\n📊 NAIVE BAYES VARIANTS COMPARISON")
    print("=" * 40)
    for variant, accuracy in results.items():
        print(f"{variant.capitalize():12}: {accuracy:.4f}")
    
    best_variant = max(results, key=results.get)
    print(f"\n🏆 Best performing variant: {best_variant.upper()} ({results[best_variant]:.4f})")

def main():
    """Main training pipeline"""
    print("🌿 TURMERIC DISEASE DETECTION - NAIVE BAYES MODEL")
    print("=" * 60)
    
    # Option 1: Train single variant
    variant = 'gaussian'  # Change to 'multinomial' or 'complement' if desired
    
    # Initialize detector
    data_path = r"E:\Turmeric dataset\Turmeric Plant Disease\Turmeric Plant Disease"
    detector = TurmericNaiveBayesDetector(data_path, nb_variant=variant)
    
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
    detector.save_model()
    
    print(f"\n🎉 {variant.upper()} NAIVE BAYES TRAINING COMPLETED!")
    print("✅ Model saved and ready for use")
    print("📱 Use 'nb_streamlit_app.py' for the web interface")
    
    # Uncomment the line below to compare all variants
    # compare_nb_variants()

if __name__ == "__main__":
    main()
