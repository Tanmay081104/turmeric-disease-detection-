import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TurmericDiseaseDetector:
    def __init__(self, data_path, img_height=224, img_width=224, batch_size=32):
        self.data_path = data_path
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        
    def prepare_data(self):
        """Prepare training and validation data generators"""
        print("Preparing data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Training data generator
        self.train_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation data generator
        self.val_generator = val_datagen.flow_from_directory(
            self.data_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Found {self.train_generator.samples} training images")
        print(f"Found {self.val_generator.samples} validation images")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        
        return self.train_generator, self.val_generator
    
    def build_model(self):
        """Build the model using transfer learning"""
        print("Building model...")
        
        # Load pre-trained MobileNetV2 model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom classifier on top
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model built successfully!")
        return self.model
    
    def train_model(self, epochs=20):
        """Train the model"""
        print("Starting training...")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'best_turmeric_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def fine_tune_model(self, epochs=10):
        """Fine-tune the model by unfreezing some layers"""
        print("Starting fine-tuning...")
        
        # Find the base model (MobileNetV2) in the model
        base_model = None
        for layer in self.model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 100:
                base_model = layer
                break
        
        if base_model is None:
            print("Could not find base model for fine-tuning. Skipping...")
            return None
        
        # Unfreeze the top layers of the base model
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before the fine_tune_at layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Re-compile with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for fine-tuning
        callbacks = [
            ModelCheckpoint(
                'best_turmeric_model_finetuned.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Continue training
        fine_tune_history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Fine-tuning completed!")
        return fine_tune_history
    
    def evaluate_model(self):
        """Evaluate the model and show results"""
        print("Evaluating model...")
        
        # Get predictions
        self.val_generator.reset()
        predictions = self.model.predict(self.val_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.val_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=self.class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def save_model_info(self):
        """Save model information for use in Streamlit app"""
        model_info = {
            'class_names': self.class_names,
            'img_height': self.img_height,
            'img_width': self.img_width,
            'num_classes': self.num_classes
        }
        
        with open('model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        
        print("Model information saved!")

def main():
    # Path to the dataset
    data_path = r"E:\Turmeric dataset\Turmeric Plant Disease\Turmeric Plant Disease"
    
    # Initialize detector
    detector = TurmericDiseaseDetector(data_path)
    
    # Prepare data
    train_gen, val_gen = detector.prepare_data()
    
    # Build model
    model = detector.build_model()
    
    # Train model
    history = detector.train_model(epochs=20)
    
    # Fine-tune model
    fine_tune_history = detector.fine_tune_model(epochs=10)
    
    # Evaluate model
    detector.evaluate_model()
    
    # Plot training history
    detector.plot_training_history()
    
    # Save model information
    detector.save_model_info()
    
    print("Training pipeline completed successfully!")
    print("Model saved as 'best_turmeric_model_finetuned.h5'")
    print("Model info saved as 'model_info.pkl'")

if __name__ == "__main__":
    main()
