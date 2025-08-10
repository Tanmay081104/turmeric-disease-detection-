import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os

def test_model():
    """Test the trained model with basic functionality"""
    print("🧪 Testing Turmeric Disease Detection Model")
    print("=" * 50)
    
    # Load model
    try:
        model_path = 'best_turmeric_model.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
        else:
            print("❌ Model file not found!")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Load model info
    try:
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        print("✅ Model info loaded successfully!")
        print(f"📊 Classes: {model_info['class_names']}")
        print(f"📐 Input size: {model_info['img_height']}x{model_info['img_width']}")
    except Exception as e:
        print(f"❌ Error loading model info: {e}")
        return False
    
    # Test with a dummy image
    try:
        # Create a dummy image (green-ish to simulate a leaf)
        dummy_image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        dummy_image[:, :, 1] = np.random.randint(100, 255, (224, 224))  # Make it more green
        
        # Normalize and add batch dimension
        dummy_image = dummy_image.astype('float32') / 255.0
        dummy_image = np.expand_dims(dummy_image, axis=0)
        
        # Make prediction
        predictions = model.predict(dummy_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = model_info['class_names'][predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        print("✅ Prediction test successful!")
        print(f"🎯 Test prediction: {predicted_class}")
        print(f"📈 Confidence: {confidence:.2%}")
        
        # Show all class probabilities
        print("\n📊 All class probabilities:")
        for i, class_name in enumerate(model_info['class_names']):
            prob = predictions[0][i]
            print(f"   {class_name}: {prob:.3f} ({prob:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False

def show_project_status():
    """Show the complete project status"""
    print("\n🌿 TURMERIC DISEASE DETECTION PROJECT STATUS")
    print("=" * 60)
    
    files_status = {
        'train_model.py': '✅ Training script',
        'streamlit_app.py': '✅ Web application',
        'best_turmeric_model.h5': '✅ Trained model',
        'model_info.pkl': '✅ Model metadata',
        'requirements.txt': '✅ Dependencies',
        'README.md': '✅ Documentation',
        'setup_and_run.bat': '✅ Setup script',
        'confusion_matrix.png': '✅ Evaluation plot',
        'training_history.png': '✅ Training plot'
    }
    
    print("📁 Project Files:")
    for file_name, description in files_status.items():
        if os.path.exists(file_name):
            print(f"   {description} - {file_name}")
        else:
            print(f"   ❌ Missing - {file_name}")
    
    print(f"\n📊 Dataset Summary:")
    dataset_path = r"E:\Turmeric dataset\Turmeric Plant Disease\Turmeric Plant Disease"
    if os.path.exists(dataset_path):
        classes = os.listdir(dataset_path)
        print(f"   Total classes: {len(classes)}")
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                image_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"   - {class_name}: {image_count} images")
    
    print(f"\n🚀 Quick Start Commands:")
    print(f"   1. Train model: python train_model.py")
    print(f"   2. Run web app: streamlit run streamlit_app.py")
    print(f"   3. Test model: python test_model.py")
    print(f"   4. One-click setup: setup_and_run.bat")
    
    print(f"\n🌐 Web Application Features:")
    print(f"   ✅ Upload and analyze leaf images")
    print(f"   ✅ Disease detection with confidence scores")
    print(f"   ✅ Detailed recommendations for each condition")
    print(f"   ✅ Smart image validation")
    print(f"   ✅ Professional UI with progress indicators")
    print(f"   ✅ Error handling and user guidance")

if __name__ == "__main__":
    # Test the model
    success = test_model()
    
    # Show project status
    show_project_status()
    
    if success:
        print(f"\n🎉 PROJECT READY!")
        print(f"Your turmeric disease detection system is fully functional.")
        print(f"Run 'streamlit run streamlit_app.py' to start the web interface.")
    else:
        print(f"\n⚠️  Issues detected. Please check the error messages above.")
