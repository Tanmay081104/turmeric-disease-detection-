# 🤖 Traditional ML Models for Turmeric Disease Detection

This folder contains traditional machine learning implementations using **Support Vector Machine (SVM)** and **Naive Bayes** classifiers for turmeric plant disease detection.

## 📊 **Available Models**

### 1. **Support Vector Machine (SVM)** 🎯
- **File**: `svm_model.py`
- **Web App**: `svm_streamlit_app.py`
- **Expected Accuracy**: 70-85%
- **Training Time**: 5-15 minutes
- **Best For**: General-purpose classification with good interpretability

### 2. **Naive Bayes** 🎲
- **File**: `naive_bayes_model.py`  
- **Web App**: `nb_streamlit_app.py`
- **Expected Accuracy**: 60-80%
- **Training Time**: 1-3 minutes
- **Variants**: Gaussian, Multinomial, Complement

## 🚀 **Quick Start**

### **Prerequisites**
```bash
# Install required packages
pip install -r requirements.txt
```

### **Training Models**

#### **Train SVM Model:**
```bash
python svm_model.py
```
- Extracts features using computer vision
- Performs hyperparameter tuning with GridSearchCV
- Applies PCA for dimensionality reduction
- Saves trained model as `svm_turmeric_model.pkl`

#### **Train Naive Bayes Models:**
```bash
python naive_bayes_model.py
```
- Trains Gaussian Naive Bayes by default
- Can train all three variants by uncommenting `compare_nb_variants()`
- Uses feature selection to choose top 50 features
- Saves models as `*_nb_turmeric_model.pkl`

### **Running Web Applications**

#### **SVM Web App:**
```bash
streamlit run svm_streamlit_app.py
```

#### **Naive Bayes Web App:**
```bash
streamlit run nb_streamlit_app.py
```

## 🔧 **Technical Details**

### **SVM Implementation**
- **Algorithm**: Support Vector Classification with RBF/Linear/Polynomial kernels
- **Features**: Color histograms, texture, edges, shape, statistical moments
- **Preprocessing**: StandardScaler + PCA (100 components)
- **Hyperparameters**: Grid search over C, kernel, gamma
- **Feature Count**: ~100+ features reduced to 100 via PCA

### **Naive Bayes Implementation**  
- **Variants**: Gaussian, Multinomial, Complement Naive Bayes
- **Features**: Color histograms, texture, shape, statistical features
- **Preprocessing**: Feature selection (top 50) + scaling
- **Feature Count**: ~100+ features reduced to 50 via SelectKBest
- **Special Handling**: NaN/infinity values cleaned, non-negative features for Multinomial/Complement

## 📈 **Feature Engineering**

Both models use comprehensive feature extraction:

### **Color Features**
- RGB color histograms (32 bins per channel)
- HSV color histograms (12-16 bins per channel)  
- Color moments (mean, std, skewness, kurtosis)
- Green color dominance detection

### **Texture Features**
- Edge density using Canny edge detection
- Local variance analysis
- Gray-level statistical measures
- Texture standard deviation and mean

### **Shape Features**
- Contour analysis
- Area and perimeter calculations
- Circularity measurements
- Bounding box properties

### **Statistical Features**
- Mean, standard deviation, variance
- Min, max, median values
- Percentiles (25th, 75th)
- Moment-based features

## 🆚 **Model Comparison**

| Aspect | SVM | Naive Bayes | Deep Learning (MobileNetV2) |
|--------|-----|-------------|------------------------------|
| **Accuracy** | 70-85% | 60-80% | 94%+ |
| **Training Time** | 5-15 min | 1-3 min | 30-60 min |
| **Inference Speed** | Very Fast | Very Fast | Fast |
| **Interpretability** | High | Very High | Low |
| **Feature Engineering** | Required | Required | Automatic |
| **Data Requirements** | Moderate | Low | High |
| **Memory Usage** | Low | Very Low | Moderate |
| **Overfitting Risk** | Moderate | Low | High (without regularization) |

## 🎯 **When to Use Each Model**

### **Use SVM When:**
- Need good balance between accuracy and speed
- Want interpretable decision boundaries  
- Have moderate amount of training data
- Computational resources are limited
- Need robust performance with outliers

### **Use Naive Bayes When:**
- Need very fast training and prediction
- Have limited training data
- Want probabilistic predictions
- Need simple, interpretable model
- Building baseline or prototype quickly

### **Use Deep Learning When:**
- Maximum accuracy is required
- Have large datasets available
- Computational resources are abundant
- Can accept "black box" predictions
- Building production systems

## 🔍 **Model Analysis**

### **SVM Strengths:**
- ✅ Good accuracy for traditional ML
- ✅ Works well with limited data
- ✅ Robust to overfitting
- ✅ Multiple kernel options
- ✅ Feature importance analysis

### **SVM Weaknesses:**
- ❌ Requires feature engineering
- ❌ Sensitive to feature scaling
- ❌ Hyperparameter tuning needed
- ❌ Can be slow with large datasets

### **Naive Bayes Strengths:**
- ✅ Very fast training/prediction
- ✅ Works with small datasets
- ✅ Provides probability estimates
- ✅ Simple and interpretable
- ✅ Handles missing features well

### **Naive Bayes Weaknesses:**
- ❌ Assumes feature independence
- ❌ Lower accuracy potential
- ❌ Sensitive to feature quality
- ❌ May struggle with complex patterns

## 📊 **Expected Performance**

Based on the turmeric dataset (1,063 images, 5 classes):

### **SVM Performance:**
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 70-85%
- **Best Classes**: Healthy conditions
- **Challenging Classes**: Disease variations

### **Naive Bayes Performance:**
- **Training Accuracy**: 70-85%  
- **Validation Accuracy**: 60-80%
- **Best Variant**: Usually Gaussian NB
- **Strength**: Fast probability estimates

## 🛠️ **Customization Options**

### **Modify SVM:**
```python
# In svm_model.py, adjust parameters:
param_grid = {
    'C': [0.1, 1, 10, 100],           # Regularization
    'kernel': ['linear', 'rbf'],       # Kernel type
    'gamma': ['scale', 'auto']         # Kernel coefficient
}
```

### **Modify Naive Bayes:**
```python
# In naive_bayes_model.py, change variant:
detector = TurmericNaiveBayesDetector(
    data_path, 
    nb_variant='gaussian'  # or 'multinomial', 'complement'
)
```

### **Feature Engineering:**
Both models allow easy feature modification:
- Add new feature extraction methods
- Modify existing feature calculations
- Adjust feature selection parameters
- Change scaling/preprocessing steps

## 📱 **Web Applications**

Both web apps provide:
- **Image Upload**: Drag & drop interface
- **Real-time Analysis**: Instant predictions
- **Confidence Scores**: Probability estimates
- **Feature Analysis**: Optional feature breakdown
- **Model Comparison**: Side-by-side results
- **Educational Info**: Algorithm explanations

## 🔬 **Research & Educational Use**

These implementations are perfect for:
- **Learning ML concepts**: Understanding traditional algorithms
- **Algorithm comparison**: Benchmarking different approaches  
- **Feature engineering**: Exploring manual feature design
- **Rapid prototyping**: Quick model development
- **Resource-constrained deployment**: Low-power applications

## 🚀 **Future Enhancements**

Potential improvements:
- **Ensemble Methods**: Combine SVM + Naive Bayes
- **Advanced Features**: GLCM, LBP, HOG descriptors
- **Auto-ML**: Automated hyperparameter optimization
- **Real-time Processing**: Webcam integration
- **Mobile Deployment**: Lightweight model versions

## 📞 **Support**

For issues with traditional ML models:
1. **Training Errors**: Check dataset paths and dependencies
2. **Low Accuracy**: Try feature engineering or hyperparameter tuning
3. **Memory Issues**: Reduce PCA components or feature selection count
4. **Speed Issues**: Use fewer features or simpler models

---

**🎓 Educational Note**: These traditional ML approaches demonstrate the evolution from manual feature engineering to automatic feature learning in deep learning. They're excellent for understanding machine learning fundamentals!
