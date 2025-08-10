import sys
import os
import time
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import MobileNetV2 model
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

# Try to import all models
try:
    from svm_model import TurmericSVMDetector
    svm_available = True
except:
    svm_available = False

try:
    from naive_bayes_model import TurmericNaiveBayesDetector  
    nb_available = True
except:
    nb_available = False

try:
    import tensorflow as tf
    mobilenet_available = True
except:
    mobilenet_available = False

def load_models():
    """Load all available trained models"""
    models = {}
    
    # Load SVM model
    if svm_available:
        try:
            svm_detector = TurmericSVMDetector("", img_size=(64, 64))
            if svm_detector.load_model('svm_turmeric_model.pkl'):
                models['SVM'] = svm_detector
                print("✅ SVM model loaded")
            else:
                print("❌ SVM model file not found")
        except Exception as e:
            print(f"❌ Error loading SVM: {e}")
    
    # Load Naive Bayes models
    if nb_available:
        nb_files = [
            ('gaussian_nb_turmeric_model.pkl', 'Gaussian NB'),
            ('multinomial_nb_turmeric_model.pkl', 'Multinomial NB'),
            ('complement_nb_turmeric_model.pkl', 'Complement NB')
        ]
        
        for model_file, model_name in nb_files:
            if os.path.exists(model_file):
                try:
                    nb_detector = TurmericNaiveBayesDetector("", img_size=(64, 64))
                    if nb_detector.load_model(model_file):
                        models[model_name] = nb_detector
                        print(f"✅ {model_name} model loaded")
                except Exception as e:
                    print(f"❌ Error loading {model_name}: {e}")
    
    # Load MobileNetV2 model
    if mobilenet_available:
        try:
            mobilenet_path = os.path.join(parent_dir, 'best_turmeric_model.h5')
            model_info_path = os.path.join(parent_dir, 'model_info.pkl')
            
            if os.path.exists(mobilenet_path) and os.path.exists(model_info_path):
                model = tf.keras.models.load_model(mobilenet_path)
                import pickle
                with open(model_info_path, 'rb') as f:
                    model_info = pickle.load(f)
                models['MobileNetV2'] = {'model': model, 'info': model_info}
                print("✅ MobileNetV2 model loaded")
            else:
                print("❌ MobileNetV2 model files not found")
        except Exception as e:
            print(f"❌ Error loading MobileNetV2: {e}")
    
    return models

def preprocess_for_mobilenet(image, target_size=(224, 224)):
    """Preprocess image for MobileNetV2"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size, Image.LANCZOS)
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def compare_models(image_path, models):
    """Compare all available models on a single image"""
    print(f"\n🔍 COMPARING MODELS ON: {os.path.basename(image_path)}")
    print("=" * 60)
    
    # Load image
    try:
        image = Image.open(image_path)
        print(f"📸 Image loaded: {image.size} pixels")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    results = {}
    
    # Test each model
    for model_name, model_data in models.items():
        print(f"\n🤖 Testing {model_name}...")
        
        try:
            start_time = time.time()
            
            if model_name == 'MobileNetV2':
                # MobileNetV2 prediction
                processed_image = preprocess_for_mobilenet(image)
                predictions = model_data['model'].predict(processed_image, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = model_data['info']['class_names'][predicted_class_idx]
                confidence = predictions[0][predicted_class_idx]
                
            elif 'NB' in model_name:
                # Naive Bayes prediction
                temp_path = "temp_comparison.jpg"
                image.save(temp_path)
                predicted_class, confidence = model_data.predict_image(temp_path, 0.3)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            elif model_name == 'SVM':
                # SVM prediction
                temp_path = "temp_comparison.jpg"
                image.save(temp_path)
                predicted_class, confidence = model_data.predict_image(temp_path, 0.3)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            inference_time = time.time() - start_time
            
            results[model_name] = {
                'prediction': predicted_class,
                'confidence': confidence,
                'time': inference_time
            }
            
            print(f"   ✅ Prediction: {predicted_class}")
            print(f"   📊 Confidence: {confidence:.2%}")  
            print(f"   ⏱️  Time: {inference_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[model_name] = {
                'prediction': 'Error',
                'confidence': 0.0,
                'time': 0.0
            }
    
    return results

def analyze_results(all_results):
    """Analyze results across multiple images"""
    print(f"\n📊 OVERALL ANALYSIS")
    print("=" * 50)
    
    # Count predictions by model
    model_stats = {}
    
    for image_results in all_results:
        for model_name, result in image_results.items():
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'predictions': [],
                    'confidences': [],
                    'times': [],
                    'errors': 0
                }
            
            if result['prediction'] != 'Error':
                model_stats[model_name]['predictions'].append(result['prediction'])
                model_stats[model_name]['confidences'].append(result['confidence'])
                model_stats[model_name]['times'].append(result['time'])
            else:
                model_stats[model_name]['errors'] += 1
    
    # Display statistics
    print("\n🎯 Performance Summary:")
    print("-" * 50)
    
    for model_name, stats in model_stats.items():
        if len(stats['confidences']) > 0:
            avg_confidence = np.mean(stats['confidences'])
            avg_time = np.mean(stats['times'])
            
            print(f"\n🤖 {model_name}:")
            print(f"   Average Confidence: {avg_confidence:.2%}")
            print(f"   Average Time: {avg_time:.3f}s")
            print(f"   Errors: {stats['errors']}")
            
            # Most common prediction
            if stats['predictions']:
                from collections import Counter
                most_common = Counter(stats['predictions']).most_common(1)[0]
                print(f"   Most Common: {most_common[0]} ({most_common[1]}x)")

def display_model_specs():
    """Display technical specifications of each model type"""
    print("\n📋 MODEL SPECIFICATIONS")
    print("=" * 60)
    
    specs = {
        "MobileNetV2 (Deep Learning)": {
            "Architecture": "Convolutional Neural Network",
            "Parameters": "~2.4M trainable",
            "Input Size": "224×224×3",
            "Feature Learning": "Automatic",
            "Expected Accuracy": "94%+",
            "Training Time": "30-60 minutes",
            "Inference Speed": "Fast",
            "Memory Usage": "Moderate",
            "Best For": "Maximum accuracy"
        },
        "SVM (Traditional ML)": {
            "Architecture": "Support Vector Machine",
            "Parameters": "~1000 support vectors",
            "Input Size": "64×64 → ~100 features",
            "Feature Learning": "Manual (computer vision)",
            "Expected Accuracy": "70-85%",
            "Training Time": "5-15 minutes",
            "Inference Speed": "Very Fast",
            "Memory Usage": "Low",
            "Best For": "Good balance, interpretability"
        },
        "Naive Bayes (Probabilistic)": {
            "Architecture": "Probabilistic Classifier",
            "Parameters": "Feature statistics",
            "Input Size": "64×64 → 50 selected features",
            "Feature Learning": "Manual + feature selection",
            "Expected Accuracy": "60-80%",
            "Training Time": "1-3 minutes",
            "Inference Speed": "Very Fast",
            "Memory Usage": "Very Low",
            "Best For": "Speed, probability estimates"
        }
    }
    
    for model_name, spec in specs.items():
        print(f"\n🔧 {model_name}")
        for key, value in spec.items():
            print(f"   {key:18}: {value}")

def main():
    print("🌿 TURMERIC DISEASE DETECTION MODEL COMPARISON")
    print("=" * 60)
    
    # Display model specifications
    display_model_specs()
    
    # Load available models
    print(f"\n⚡ LOADING MODELS")
    print("-" * 30)
    
    models = load_models()
    
    if not models:
        print("❌ No models available for comparison!")
        print("💡 Please train at least one model first:")
        print("   • Deep Learning: python train_model.py")
        print("   • SVM: python svm_model.py") 
        print("   • Naive Bayes: python naive_bayes_model.py")
        return
    
    print(f"\n✅ Loaded {len(models)} models: {', '.join(models.keys())}")
    
    # Test with sample images from dataset
    sample_images = []
    dataset_path = r"E:\Turmeric dataset\Turmeric Plant Disease\Turmeric Plant Disease"
    
    if os.path.exists(dataset_path):
        print(f"\n🖼️  FINDING SAMPLE IMAGES")
        print("-" * 30)
        
        # Get one image from each class
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    sample_image = os.path.join(class_path, images[0])
                    sample_images.append(sample_image)
                    print(f"   📸 {class_name}: {images[0]}")
    
    if not sample_images:
        print("❌ No sample images found!")
        return
    
    # Compare models on sample images
    print(f"\n🔍 RUNNING COMPARISONS")
    print("-" * 30)
    
    all_results = []
    
    for image_path in sample_images[:3]:  # Test first 3 images
        results = compare_models(image_path, models)
        all_results.append(results)
    
    # Analyze overall results
    if all_results:
        analyze_results(all_results)
    
    # Final recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 30)
    
    print("🎯 Choose based on your needs:")
    print("   • Maximum Accuracy → MobileNetV2")
    print("   • Best Balance → SVM") 
    print("   • Fastest Training → Naive Bayes")
    print("   • Interpretability → SVM or Naive Bayes")
    print("   • Limited Resources → Naive Bayes")
    print("   • Production System → MobileNetV2")

if __name__ == "__main__":
    main()
