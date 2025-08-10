import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pickle
import cv2
import io
import os

# Set page config
st.set_page_config(
    page_title="Turmeric Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #2E8B57;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #556B2F;
    margin-bottom: 1rem;
}
.result-box {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.healthy {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.diseased {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
}
.error {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
}
.info-box {
    background-color: #e7f3ff;
    border: 1px solid #b3d9ff;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and model information"""
    try:
        # Load model
        model_path = 'best_turmeric_model_finetuned.h5'
        if not os.path.exists(model_path):
            model_path = 'best_turmeric_model.h5'
        
        if not os.path.exists(model_path):
            st.error("❌ Model file not found! Please train the model first.")
            return None, None
        
        model = tf.keras.models.load_model(model_path)
        
        # Load model info
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        return model, model_info
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to array and normalize
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def is_leaf_image(image):
    """Simple check to determine if image might be a leaf"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to HSV for better color analysis
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define green color range in HSV
        lower_green = np.array([35, 25, 25])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green colors
        green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
        
        # Calculate the percentage of green pixels
        green_percentage = np.sum(green_mask > 0) / (img_array.shape[0] * img_array.shape[1])
        
        # If less than 10% green, probably not a leaf
        if green_percentage < 0.1:
            return False, green_percentage
        
        return True, green_percentage
    except Exception as e:
        # If error in processing, assume it might be a leaf
        return True, 0.0

def get_disease_info(class_name):
    """Get information about the detected disease/condition"""
    disease_info = {
        "Healthy Leaf": {
            "description": "The leaf appears to be healthy with no visible signs of disease.",
            "recommendations": [
                "Continue with regular care and monitoring",
                "Maintain proper watering and fertilization",
                "Ensure good air circulation around plants"
            ],
            "severity": "None"
        },
        "Dry Leaf": {
            "description": "The leaf shows signs of drying, which could be due to water stress, nutrient deficiency, or natural aging.",
            "recommendations": [
                "Check soil moisture levels",
                "Ensure adequate but not excessive watering",
                "Consider fertilization if nutrient deficiency is suspected",
                "Remove severely dried leaves to prevent spread"
            ],
            "severity": "Moderate"
        },
        "Leaf Blotch": {
            "description": "Leaf blotch is a fungal disease that causes irregular spots or blotches on leaves.",
            "recommendations": [
                "Improve air circulation around plants",
                "Avoid overhead watering",
                "Apply appropriate fungicide treatment",
                "Remove and destroy affected leaves",
                "Ensure proper plant spacing"
            ],
            "severity": "High"
        },
        "Rhizome Disease Root": {
            "description": "The rhizome (underground stem) shows signs of disease, which can affect the entire plant.",
            "recommendations": [
                "Improve soil drainage",
                "Avoid overwatering",
                "Consider soil treatment with appropriate fungicides",
                "Remove and destroy affected plant parts",
                "Consult agricultural extension services for specific treatment"
            ],
            "severity": "Very High"
        },
        "Rhizome Healthy Root": {
            "description": "The rhizome appears healthy with no visible signs of disease.",
            "recommendations": [
                "Maintain current growing conditions",
                "Continue monitoring for any changes",
                "Ensure proper soil drainage and fertility"
            ],
            "severity": "None"
        }
    }
    
    return disease_info.get(class_name, {
        "description": "Unknown condition detected.",
        "recommendations": ["Consult with agricultural experts for proper diagnosis"],
        "severity": "Unknown"
    })

def main():
    # Header
    st.markdown('<h1 class="main-header">🌿 Turmeric Leaf Disease Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, model_info = load_model()
    
    if model is None or model_info is None:
        st.error("❌ Unable to load the model. Please ensure the model files are present.")
        st.info("📝 To train the model, run: `python train_model.py`")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🔧 Settings</h2>', unsafe_allow_html=True)
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence required for prediction"
        )
        
        show_probabilities = st.checkbox("Show All Probabilities", value=False)
        
        st.markdown('<h2 class="sub-header">📊 Model Info</h2>', unsafe_allow_html=True)
        st.info(f"""
        **Classes:** {', '.join(model_info['class_names'])}
        
        **Input Size:** {model_info['img_height']}x{model_info['img_width']}
        
        **Total Classes:** {model_info['num_classes']}
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a turmeric leaf or rhizome"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Check if image might be a leaf
            is_leaf, green_percentage = is_leaf_image(image)
            
            if not is_leaf:
                st.markdown("""
                <div class="result-box error">
                    <h3>⚠️ Warning: This might not be a leaf image</h3>
                    <p>The uploaded image has very little green content ({:.1%} green pixels). 
                    Please upload a clear image of a turmeric leaf or rhizome for accurate detection.</p>
                </div>
                """.format(green_percentage), unsafe_allow_html=True)
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(
                        image, 
                        target_size=(model_info['img_height'], model_info['img_width'])
                    )
                    
                    if processed_image is not None:
                        # Make prediction
                        predictions = model.predict(processed_image)
                        predicted_class_idx = np.argmax(predictions[0])
                        predicted_class = model_info['class_names'][predicted_class_idx]
                        confidence = predictions[0][predicted_class_idx]
                        
                        # Display results in second column
                        with col2:
                            st.markdown('<h2 class="sub-header">📋 Analysis Results</h2>', unsafe_allow_html=True)
                            
                            # Main prediction result
                            if confidence >= confidence_threshold:
                                disease_info = get_disease_info(predicted_class)
                                
                                # Determine result style based on severity
                                if disease_info['severity'] in ['None']:
                                    result_class = "healthy"
                                    icon = "✅"
                                elif disease_info['severity'] in ['Moderate']:
                                    result_class = "diseased"
                                    icon = "⚠️"
                                else:
                                    result_class = "diseased"
                                    icon = "🚨"
                                
                                st.markdown(f"""
                                <div class="result-box {result_class}">
                                    <h3>{icon} Prediction: {predicted_class}</h3>
                                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                    <p><strong>Severity:</strong> {disease_info['severity']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Disease information
                                with st.expander("📖 Detailed Information", expanded=True):
                                    st.write("**Description:**")
                                    st.write(disease_info['description'])
                                    
                                    st.write("**Recommendations:**")
                                    for i, rec in enumerate(disease_info['recommendations'], 1):
                                        st.write(f"{i}. {rec}")
                                
                            else:
                                st.markdown(f"""
                                <div class="result-box error">
                                    <h3>❓ Low Confidence Prediction</h3>
                                    <p><strong>Best Guess:</strong> {predicted_class}</p>
                                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                    <p>The model is not confident about this prediction. Please try uploading a clearer image.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show all probabilities if requested
                            if show_probabilities:
                                st.markdown("### 📊 All Class Probabilities")
                                prob_data = []
                                for i, class_name in enumerate(model_info['class_names']):
                                    prob_data.append({
                                        'Class': class_name,
                                        'Probability': predictions[0][i],
                                        'Percentage': f"{predictions[0][i]:.2%}"
                                    })
                                
                                # Sort by probability
                                prob_data.sort(key=lambda x: x['Probability'], reverse=True)
                                
                                for item in prob_data:
                                    st.write(f"**{item['Class']}:** {item['Percentage']}")
                                    st.progress(float(item['Probability']))
    
    # Information section
    with st.expander("ℹ️ About This Application", expanded=False):
        st.markdown("""
        ### About Turmeric Leaf Disease Detection
        
        This application uses deep learning to identify common diseases and conditions in turmeric plants. 
        It can detect the following conditions:
        
        - **Healthy Leaf**: Normal, healthy turmeric leaves
        - **Dry Leaf**: Leaves showing signs of drying or water stress
        - **Leaf Blotch**: Fungal disease causing spots on leaves
        - **Rhizome Disease Root**: Diseased underground stems/roots
        - **Rhizome Healthy Root**: Healthy underground stems/roots
        
        ### How to Use
        1. Upload a clear image of a turmeric leaf or rhizome
        2. Click "Analyze Image" to get predictions
        3. Review the results and recommendations
        
        ### Tips for Best Results
        - Use clear, well-lit images
        - Ensure the leaf/rhizome fills most of the frame
        - Avoid blurry or low-quality images
        - Take photos against a neutral background when possible
        
        ### Disclaimer
        This tool is for educational and guidance purposes only. For serious plant health issues, 
        please consult with agricultural experts or extension services.
        """)

if __name__ == "__main__":
    main()
