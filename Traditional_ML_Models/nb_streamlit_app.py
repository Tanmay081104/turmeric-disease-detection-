import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import sys
import pickle

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))
from naive_bayes_model import TurmericNaiveBayesDetector

# Set page config
st.set_page_config(
    page_title="Turmeric Disease Detection - Naive Bayes",
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
    color: #FF6B35;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #D84315;
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
.nb-info {
    background-color: #f3e5f5;
    border: 1px solid #e1bee7;
    color: #4a148c;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nb_models():
    """Load all available Naive Bayes models"""
    models = {}
    model_files = ['gaussian_nb_turmeric_model.pkl', 'multinomial_nb_turmeric_model.pkl', 'complement_nb_turmeric_model.pkl']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                detector = TurmericNaiveBayesDetector("", img_size=(64, 64))
                success = detector.load_model(model_file)
                if success:
                    variant_name = model_file.split('_')[0]
                    models[variant_name.title()] = detector
            except Exception as e:
                st.warning(f"Could not load {model_file}: {e}")
    
    return models

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
    st.markdown('<h1 class="main-header">🎲 Naive Bayes Turmeric Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Probabilistic Machine Learning Approach using Naive Bayes</p>', unsafe_allow_html=True)
    
    # Load models
    models = load_nb_models()
    
    if not models:
        st.error("❌ No Naive Bayes models found! Please train the models first.")
        st.info("📝 To train Naive Bayes models, run: `python naive_bayes_model.py`")
        
        # Show model info if available
        st.markdown("### 📊 About Naive Bayes Models")
        st.markdown("""
        **Naive Bayes** is a probabilistic machine learning algorithm that:
        - Assumes feature independence (naive assumption)
        - Uses Bayes' theorem for classification
        - Works well with categorical and continuous features
        - Requires minimal training data
        - Provides probability estimates
        
        **Three Variants Available:**
        - **Gaussian NB**: For continuous features (normal distribution)
        - **Multinomial NB**: For discrete/count features
        - **Complement NB**: Better for imbalanced datasets
        
        **Expected Performance**: 60-80% accuracy (depending on feature quality)
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🎛️ Naive Bayes Settings</h2>', unsafe_allow_html=True)
        
        # Model selection
        selected_model = st.selectbox(
            "Select Naive Bayes Variant",
            list(models.keys()),
            help="Choose which Naive Bayes variant to use for prediction"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.2,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Minimum confidence required for prediction"
        )
        
        show_probabilities = st.checkbox("Show All Class Probabilities", value=True)
        
        st.markdown('<h2 class="sub-header">📊 Model Info</h2>', unsafe_allow_html=True)
        
        if selected_model in models:
            detector = models[selected_model]
            model_info = f"""
            **Algorithm**: {selected_model} Naive Bayes
            **Classes**: {len(detector.class_names)}
            **Image Size**: {detector.img_size[0]}×{detector.img_size[1]}
            **Feature Selection**: Top 50 features
            **Variant**: {detector.nb_variant.title()}
            """
            st.info(model_info)
        
        # Show advantages of each variant
        st.markdown("### 🎯 NB Variant Comparison")
        variant_info = {
            "Gaussian": "📊 Best for continuous features with normal distribution",
            "Multinomial": "🔢 Good for discrete/count features (word counts, histograms)",
            "Complement": "⚖️ Better handling of imbalanced datasets"
        }
        
        for variant, description in variant_info.items():
            if variant in models:
                st.success(f"✅ {variant}: {description}")
            else:
                st.error(f"❌ {variant}: Not available")
    
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
            if st.button("🎲 Analyze with Naive Bayes", type="primary"):
                if selected_model not in models:
                    st.error(f"❌ {selected_model} Naive Bayes model not loaded!")
                    return
                
                detector = models[selected_model]
                
                with st.spinner(f"Analyzing image using {selected_model} Naive Bayes..."):
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
                            st.markdown('<h2 class="sub-header">🎯 Naive Bayes Results</h2>', unsafe_allow_html=True)
                            
                            # Show model variant being used
                            st.markdown(f"""
                            <div class="nb-info">
                                <strong>🎲 Using:</strong> {selected_model} Naive Bayes<br>
                                <strong>📊 Algorithm:</strong> Probabilistic Classification
                            </div>
                            """, unsafe_allow_html=True)
                            
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
                                
                                # Show probabilities
                                if show_probabilities:
                                    st.markdown("### 🎲 Class Probabilities")
                                    
                                    # Get all probabilities
                                    features = detector.extract_features(temp_path if os.path.exists(temp_path) else uploaded_file)
                                    if features is not None:
                                        # Preprocess features
                                        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
                                        
                                        if detector.nb_variant == 'gaussian':
                                            features_scaled = detector.scaler.transform([features])
                                        else:
                                            features_scaled = detector.scaler.transform([np.abs(features)])
                                        
                                        features_selected = detector.feature_selector.transform(features_scaled)
                                        probabilities = detector.model.predict_proba(features_selected)[0]
                                        
                                        # Create probability chart
                                        prob_data = []
                                        for i, class_name in enumerate(detector.class_names):
                                            prob_data.append({
                                                'class': class_name,
                                                'probability': probabilities[i],
                                                'percentage': f"{probabilities[i]:.1%}"
                                            })
                                        
                                        # Sort by probability
                                        prob_data.sort(key=lambda x: x['probability'], reverse=True)
                                        
                                        for item in prob_data:
                                            col2_1, col2_2, col2_3 = st.columns([2, 1, 1])
                                            with col2_1:
                                                st.write(f"**{item['class']}**")
                                            with col2_2:
                                                st.write(item['percentage'])
                                            with col2_3:
                                                st.progress(float(item['probability']))
                                
                            else:
                                st.markdown(f"""
                                <div class="result-box error">
                                    <h3>❓ Low Confidence Prediction</h3>
                                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                    <p>The Naive Bayes model is not confident about this prediction. This could be due to:</p>
                                    <ul>
                                        <li>Feature values not matching training patterns</li>
                                        <li>Poor image quality or unusual lighting</li>
                                        <li>Image not containing a turmeric leaf</li>
                                        <li>Need for more training data</li>
                                    </ul>
                                    <p>Try uploading a clearer image or adjusting the confidence threshold.</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during Naive Bayes analysis: {str(e)}")
    
    # Model comparison section
    if len(models) > 1:
        st.markdown("---")
        st.markdown("### 🔬 Model Comparison")
        
        if st.button("🏆 Compare All Naive Bayes Variants"):
            if uploaded_file is not None:
                comparison_results = {}
                
                # Save temp image
                temp_path = "temp_comparison.jpg"
                image.save(temp_path)
                
                for model_name, detector in models.items():
                    try:
                        predicted_class, confidence = detector.predict_image(temp_path, 0.3)  # Lower threshold for comparison
                        comparison_results[model_name] = {
                            'prediction': predicted_class,
                            'confidence': confidence
                        }
                    except Exception as e:
                        comparison_results[model_name] = {
                            'prediction': 'Error',
                            'confidence': 0.0
                        }
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Display comparison
                st.markdown("#### 📊 Comparison Results:")
                
                col_comp1, col_comp2, col_comp3 = st.columns(3)
                cols = [col_comp1, col_comp2, col_comp3]
                
                for i, (model_name, result) in enumerate(comparison_results.items()):
                    with cols[i % 3]:
                        st.metric(
                            label=f"🎲 {model_name} NB",
                            value=result['prediction'],
                            delta=f"{result['confidence']:.1%}"
                        )
            else:
                st.warning("Please upload an image first to compare models.")
    
    # Information section
    with st.expander("ℹ️ About Naive Bayes Classification", expanded=False):
        st.markdown("""
        ### 🎲 Naive Bayes Approach
        
        **Naive Bayes** is a probabilistic machine learning algorithm based on **Bayes' Theorem**:
        
        ```
        P(Class|Features) = P(Features|Class) × P(Class) / P(Features)
        ```
        
        #### 🧠 How It Works:
        1. **Feature Extraction**: Extract color, texture, and shape features
        2. **Feature Selection**: Choose most discriminative features (top 50)
        3. **Probability Calculation**: Estimate probability for each class
        4. **Classification**: Choose class with highest probability
        
        #### 🔄 Three Variants Available:
        
        **1. Gaussian Naive Bayes** 📊
        - Assumes features follow normal distribution
        - Best for continuous numerical features
        - Good general-purpose classifier
        
        **2. Multinomial Naive Bayes** 🔢
        - For discrete/count features
        - Works well with histogram features
        - Good for sparse feature vectors
        
        **3. Complement Naive Bayes** ⚖️
        - Adaptation for imbalanced datasets
        - Reduces bias toward frequent classes
        - Often performs better than multinomial
        
        #### ⚡ Advantages:
        - **Fast**: Very quick training and prediction
        - **Simple**: Easy to understand and implement
        - **Probabilistic**: Provides confidence estimates
        - **Robust**: Handles missing features well
        - **Low Data**: Works with small training sets
        
        #### ⚠️ Limitations:
        - **Naive Assumption**: Assumes feature independence
        - **Feature Quality**: Depends heavily on good features
        - **Accuracy**: May not match sophisticated algorithms
        
        ### 🆚 Comparison Table:
        
        | Aspect | Gaussian NB | Multinomial NB | Complement NB |
        |--------|-------------|----------------|---------------|
        | **Best For** | Continuous features | Count/discrete data | Imbalanced classes |
        | **Assumption** | Normal distribution | Multinomial distribution | Modified multinomial |
        | **Performance** | Good | Moderate | Often best for text |
        | **Speed** | Very fast | Very fast | Very fast |
        
        ### 🎯 When to Use Naive Bayes:
        - **Baseline Model**: Quick initial assessment
        - **Real-time Applications**: Fast prediction needed
        - **Limited Data**: Small training datasets
        - **Interpretability**: Need to understand probabilities
        - **Educational**: Learning ML concepts
        
        ### 💡 Tips for Better Results:
        - Use feature selection to remove irrelevant features
        - Try different variants to see which works best
        - Consider ensemble methods combining multiple variants
        - Adjust confidence thresholds based on application needs
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🎲 **Probabilistic ML Approach** | Powered by Naive Bayes & Feature Engineering")

if __name__ == "__main__":
    main()
