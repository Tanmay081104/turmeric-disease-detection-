# 🌿 Turmeric Leaf Disease Detection System

A deep learning-based system for detecting diseases in turmeric plants using computer vision. The system can identify 5 different conditions: Healthy Leaf, Dry Leaf, Leaf Blotch, Rhizome Disease Root, and Rhizome Healthy Root.

## 📋 Features

- **Deep Learning Model**: Uses MobileNetV2 with transfer learning for accurate disease detection
- **Web Interface**: User-friendly Streamlit app for easy image upload and analysis
- **Multiple Disease Detection**: Identifies 5 different plant conditions
- **Confidence Scoring**: Provides confidence levels for predictions
- **Detailed Recommendations**: Offers specific care recommendations for each detected condition
- **Image Validation**: Basic checks to ensure uploaded images are likely to be plant images
- **Real-time Analysis**: Fast prediction and results display

## 🏗️ Project Structure

```
Turmeric dataset/
├── train_model.py          # Model training script
├── streamlit_app.py        # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── Turmeric Plant Disease/
│   └── Turmeric Plant Disease/
│       ├── Dry Leaf/       # Dry leaf images
│       ├── Healthy Leaf/   # Healthy leaf images
│       ├── Leaf Blotch/    # Leaf blotch images
│       ├── Rhizome Disease Root/  # Diseased root images
│       └── Rhizome Healthy Root/  # Healthy root images
└── (Generated files after training)
    ├── best_turmeric_model_finetuned.h5  # Trained model
    ├── model_info.pkl                    # Model metadata
    ├── confusion_matrix.png              # Model evaluation results
    └── training_history.png              # Training progress charts
```

## 🚀 Quick Start

### 1. Install Dependencies

First, make sure you have Python 3.8+ installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the training script to create and train the disease detection model:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Create a MobileNetV2-based model
- Train the model with data augmentation
- Fine-tune the model for better performance
- Save the trained model and metadata
- Generate evaluation plots

**Training time**: Approximately 30-60 minutes (depending on your hardware)

### 3. Launch the Web Application

After training is complete, start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The app will open in your web browser at `http://localhost:8501`

## 📱 Using the Web Application

1. **Upload Image**: Click "Choose an image file" and select a turmeric leaf or rhizome image
2. **Analyze**: Click the "🔍 Analyze Image" button
3. **Review Results**: View the prediction, confidence score, and recommendations
4. **Adjust Settings**: Use the sidebar to modify confidence thresholds and view options

### Supported Image Formats
- PNG (.png)
- JPEG (.jpg, .jpeg)

### Tips for Best Results
- Use clear, well-lit images
- Ensure the leaf/rhizome fills most of the frame
- Avoid blurry or low-quality images
- Take photos against a neutral background when possible

## 🎯 Model Performance

The system can detect 5 different conditions:

1. **Healthy Leaf** - Normal, healthy turmeric leaves
2. **Dry Leaf** - Leaves showing signs of drying or water stress
3. **Leaf Blotch** - Fungal disease causing spots on leaves
4. **Rhizome Disease Root** - Diseased underground stems/roots
5. **Rhizome Healthy Root** - Healthy underground stems/roots

## 🔧 Technical Details

### Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**: Global Average Pooling + Dense layers with dropout
- **Input Size**: 224x224 pixels
- **Output**: 5-class classification with softmax activation

### Training Strategy
1. **Transfer Learning**: Start with frozen MobileNetV2 features
2. **Data Augmentation**: Rotation, shifts, flips, zoom, shear
3. **Initial Training**: Train custom classifier layers
4. **Fine-tuning**: Unfreeze and train top layers with lower learning rate

### Key Features
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Model Checkpointing**: Saves best model during training
- **Comprehensive Evaluation**: Classification report and confusion matrix

## 📊 Dataset Information

Your dataset contains:
- **Dry Leaf**: 203 images
- **Healthy Leaf**: 197 images
- **Leaf Blotch**: 199 images
- **Rhizome Disease Root**: 182 images
- **Rhizome Healthy Root**: 282 images

**Total**: 1,063 images across 5 classes

## 🛠️ Customization

### Modifying the Model
Edit `train_model.py` to:
- Change model architecture
- Adjust hyperparameters
- Modify data augmentation
- Add new classes

### Customizing the Web App
Edit `streamlit_app.py` to:
- Change UI layout
- Modify disease information
- Add new features
- Customize styling

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.10+
- Streamlit 1.28+
- OpenCV 4.5+
- Other dependencies listed in `requirements.txt`

## ⚠️ Important Notes

### Error Handling
- The app includes image validation to detect non-plant images
- Confidence thresholds help filter low-quality predictions
- Error messages guide users when issues occur

### Limitations
- Model performance depends on image quality
- Trained specifically for turmeric plants
- May not generalize to other plant species
- Should not replace professional agricultural advice

## 🎓 Educational Purpose

This system is designed for:
- Learning about plant disease detection
- Understanding deep learning applications in agriculture
- Demonstrating transfer learning techniques
- Exploring computer vision for practical problems

## 📞 Support

If you encounter issues:

1. **Training Problems**: Ensure all dependencies are installed and dataset path is correct
2. **App Not Loading**: Check that the model files exist after training
3. **Prediction Issues**: Try uploading clearer, well-lit images
4. **Performance Issues**: Consider reducing image size or batch size for slower hardware

## 🔮 Future Improvements

Potential enhancements:
- Mobile app development
- Real-time camera integration
- Additional plant species support
- Treatment recommendation system
- Integration with agricultural databases
- Multi-language support

## 📄 License

This project is for educational purposes. Please ensure you have appropriate rights to use the dataset.

## 🙏 Disclaimer

This tool is for educational and guidance purposes only. For serious plant health issues, please consult with agricultural experts or extension services.

---

Happy plant disease detection! 🌱
