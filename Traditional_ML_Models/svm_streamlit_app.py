import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import sys
import pickle

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))
from svm_model import TurmericSVMDetector

# Set page config
st.set_page_config(
    page_title="Turmeric Disease Detection - SVM",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #4CAF50;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #2E7D32;
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
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_svm_model():
    """Load the trained SVM model"""
    try:
        detector = TurmericSVMDetector("", img_size=(64, 64))
        success = detector.load_model('svm_turmeric_model.pkl')
        if success:
            return detector
        else:
            return None
    except Exception as e:
        st.error(f"Error loading SVM model: {e}")
        return None

def is_leaf_image(image):
    """Simple check to determine if image might be a leaf"""
    try:
        img_array = np.array(image)
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define green color range
        lower_green = np.array([35, 25, 25])
        upper_green = np.array([85, 255, 255])
        
        green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
        green_percentage = np.sum(green_mask > 0) / (img_array.shape[0] * img_array.shape[1])
        
        if green_percentage < 0.08:
            return False, green_percentage
        return True, green_percentage
    except:
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
            "severity": "None",
            "icon": "✅"
        },
        "Dry Leaf": {
            "description": "The leaf shows signs of drying, which could be due to water stress, nutrient deficiency, or natural aging.",
            "recommendations": [
                "Check soil moisture levels",
                "Ensure adequate but not excessive watering",
                "Consider fertilization if nutrient deficiency is suspected",
                "Remove severely dried leaves to prevent spread"
            ],
            "severity": "Moderate",
            "icon": "⚠️"
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
            "severity": "High",
            "icon": "🚨"
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
            "severity": "Very High",
            "icon": "🚨"
        },
        "Rhizome Healthy Root": {
            "description": "The rhizome appears healthy with no visible signs of disease.",
            "recommendations": [
                "Maintain current growing conditions",
                "Continue monitoring for any changes",
                "Ensure proper soil drainage and fertility"
            ],
            "severity": "None",
            "icon": "✅"
        }
    }
    
    return disease_info.get(class_name, {
        "description": "Unknown condition detected.",
        "recommendations": ["Consult with agricultural experts for proper diagnosis"],
        "severity": "Unknown",
        "icon": "❓"
    })

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 SVM-Based Turmeric Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Traditional Machine Learning Approach using Support Vector Machine</p>', unsafe_allow_html=True)
    
    # Load model
    detector = load_svm_model()
    
    if detector is None:
        st.error("❌ SVM model not found! Please train the model first.")
        st.info("📝 To train the SVM model, run: `python svm_model.py`")
        
        # Show model info if available
        st.markdown("### 📊 About SVM Model")
        st.markdown("""
        **Support Vector Machine (SVM)** is a traditional machine learning algorithm that:
        - Uses handcrafted features (color histograms, texture, edges, shape)
        - Finds optimal decision boundaries between classes
        - Works well with smaller datasets
        - Provides interpretable results
        - Trains faster than deep learning models
        
        **Expected Performance**: 70-85% accuracy (depending on feature quality)
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ SVM Settings</h2>', unsafe_allow_html=True)
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.3,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence required for prediction"
        )
        
        show_features = st.checkbox("Show Feature Analysis", value=False)
        
        st.markdown('<h2 class="sub-header">📊 Model Info</h2>', unsafe_allow_html=True)
        
        # Model specifications
        model_info = f"""
        **Algorithm**: Support Vector Machine
        **Kernel**: {detector.model.kernel if detector.model else 'Unknown'}
        **Classes**: {len(detector.class_names)}
        **Image Size**: {detector.img_size[0]}×{detector.img_size[1]}
        **Features**: Extracted using computer vision
        """
        st.info(model_info)
        
        # Performance metrics (if available)
        st.markdown("### 🎯 SVM Advantages")
        st.markdown("""
        - ⚡ Fast training and inference
        - 🔍 Works with limited data  
        - 📊 Interpretable decision boundaries
        - 🛠️ Robust to outliers
        - 💰 Low computational requirements
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
            
            # Image validation
            is_leaf, green_percentage = is_leaf_image(image)
            
            if not is_leaf:
                st.warning(f"⚠️ This image has very little green content ({green_percentage:.1%}). Please upload a leaf image for better results.")
            
            # Analyze button
            if st.button("🔍 Analyze with SVM", type="primary"):
                with st.spinner("Analyzing image using SVM..."):
                    try:
                        # Save temporary image for processing
                        temp_path = "temp_image.jpg"
                        image.save(temp_path)
                        
                        # Make prediction
                        predicted_class, confidence = detector.predict_image(temp_path, confidence_threshold)
                        
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        # Display results
                        with col2:
                            st.markdown('<h2 class="sub-header">🎯 SVM Analysis Results</h2>', unsafe_allow_html=True)
                            
                            if predicted_class and predicted_class != "Low Confidence":
                                disease_info = get_disease_info(predicted_class)
                                
                                # Result styling
                                if disease_info['severity'] in ['None']:
                                    result_class = "healthy"
                                else:
                                    result_class = "diseased"
                                
                                st.markdown(f"""
                                <div class="result-box {result_class}">
                                    <h3>{disease_info['icon']} Prediction: {predicted_class}</h3>
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
                                
                                # Feature analysis
                                if show_features:
                                    st.markdown("### 🔬 Feature Analysis")
                                    
                                    # Extract and display some features
                                    features = detector.extract_features(temp_path if os.path.exists(temp_path) else None)
                                    if features is not None:
                                        col2_1, col2_2 = st.columns(2)
                                        
                                        with col2_1:
                                            st.metric("Color Features", f"{len([f for f in features[:96]])}")
                                            st.metric("Texture Features", f"{len([f for f in features[96:98]])}")
                                        
                                        with col2_2:
                                            st.metric("Shape Features", f"{len([f for f in features[98:101]])}")
                                            st.metric("Total Features", f"{len(features)}")
                                
                            else:
                                st.markdown(f"""
                                <div class="result-box error">
                                    <h3>❓ Low Confidence Prediction</h3>
                                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                    <p>The SVM model is not confident about this prediction. This could be due to:</p>
                                    <ul>
                                        <li>Poor image quality or lighting</li>
                                        <li>Unusual leaf appearance</li>
                                        <li>Image not containing a turmeric leaf</li>
                                    </ul>
                                    <p>Try uploading a clearer, well-lit image of a turmeric leaf.</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during SVM analysis: {str(e)}")
    
    # Information section
    with st.expander("ℹ️ About SVM-Based Detection", expanded=False):
        st.markdown("""
        ### 🤖 Support Vector Machine Approach
        
        This system uses **Support Vector Machine (SVM)** - a traditional machine learning algorithm that works by:
        
        #### 🔧 Feature Extraction Process:
        1. **Color Analysis**: RGB and HSV color histograms
        2. **Texture Analysis**: Local variance and edge detection
        3. **Shape Analysis**: Contour-based features
        4. **Statistical Features**: Mean, standard deviation, percentiles
        
        #### 📊 How SVM Works:
        - Finds optimal decision boundaries between different disease classes
        - Uses kernel functions to handle non-linear relationships
        - Makes predictions based on distance from decision boundaries
        
        #### ⚡ Advantages:
        - **Fast**: Quick training and inference
        - **Efficient**: Works well with smaller datasets
        - **Robust**: Less prone to overfitting
        - **Interpretable**: Can analyze feature importance
        
        #### ⚠️ Limitations:
        - **Feature Engineering**: Relies on handcrafted features
        - **Lower Accuracy**: May not achieve deep learning performance
        - **Scale Sensitivity**: Requires proper feature scaling
        
        ### 🆚 Comparison with Deep Learning:
        | Aspect | SVM | Deep Learning (MobileNetV2) |
        |--------|-----|------------------------------|
        | Accuracy | 70-85% | 94%+ |
        | Training Time | Minutes | Hours |
        | Data Required | Moderate | Large |
        | Interpretability | High | Low |
        | Resource Usage | Low | High |
        
        ### 💡 Best Use Cases:
        - Quick prototyping and testing
        - Limited computational resources
        - Small to medium datasets
        - When interpretability is important
        - Educational purposes
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🌱 **Traditional ML Approach** | Powered by SVM & Computer Vision")

if __name__ == "__main__":
    main()
