import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.layers import (Reshape, Multiply, Add, GlobalAvgPool2D, 
                                   DepthwiseConv2D, BatchNormalization, 
                                   Activation, Dropout, Dense, Conv2D)
from tensorflow.keras.optimizers.schedules import CosineDecay
import albumentations as A
from tensorflow.keras.mixed_precision import global_policy
import tensorflow.keras.backend as K

# Configure GPU memory - fixed allocation instead of growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Reserve only 768MB VRAM
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=768)]
        )
    except RuntimeError as e:
        print(e)

# Device placement strategy
cpu_device = tf.device('/CPU:0')
gpu_device = tf.device('/GPU:0')

# ========== DATA LOADING AND PADDING ==========
def pad_image(image, target_size=(150, 150)):
    """
    Pads or resizes the image to match the target size.
    Args:
    - image: Input image (H, W, C)
    - target_size: Tuple (height, width) specifying desired output size
    
    Returns:
    - Padded or resized image of shape (target_size[0], target_size[1], C)
    """
    h, w, c = image.shape
    target_h, target_w = target_size
    
    # If the image is larger, resize it directly
    if h > target_h or w > target_w:
        return cv2.resize(image, (target_w, target_h))
    
    # Compute padding values
    delta_h = target_h - h
    delta_w = target_w - w
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    
    # Apply padding
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                      borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def load_dataset_with_padding(dataset_path, target_size=(150, 150)):
    """
    Loads images from a dataset directory, pads/resizes them, and returns arrays of images and labels.
    Args:
    - dataset_path: Path to the dataset directory
    - target_size: Tuple (height, width) specifying desired output size
    
    Returns:
    - images: Numpy array of padded/resized images
    - labels: Numpy array of corresponding labels
    """
    images = []
    labels = []
    
    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                if image_path.endswith(('.jpg', '.png')):
                    image = cv2.imread(image_path)
                    if image is not None:  # Ensure the image is loaded correctly
                        padded_image = pad_image(image, target_size)
                        images.append(padded_image)
                        labels.append(class_name)
    
    return np.array(images), np.array(labels)

class ProgressiveLoader:
    def __init__(self, dataset_path, batch_size=16, chunk_size=1000):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.current_chunk = 0
        self.total_samples = sum(len(files) for _, _, files in os.walk(dataset_path))
        
    def load_chunk(self):
        with cpu_device:
            start_idx = self.current_chunk * self.chunk_size
            if start_idx >= self.total_samples:
                return None, None
            
            images, labels = [], []
            count = 0
            
            for class_name in os.listdir(self.dataset_path):
                if count >= self.chunk_size:
                    break
                    
                class_path = os.path.join(self.dataset_path, class_name)
                if os.path.isdir(class_path):
                    files = os.listdir(class_path)[start_idx:start_idx + self.chunk_size]
                    for f in files:
                        if count >= self.chunk_size:
                            break
                        img_path = os.path.join(class_path, f)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = pad_image(img, (128, 128))  # Reduced size
                            images.append(img)
                            labels.append(class_name)
                            count += 1
            
            self.current_chunk += 1
            return np.array(images), np.array(labels)

# ========== FUCKING EFFICIENT DATA PIPELINE ==========
# Create augmentation layers at module level
AUGMENTATION_LAYERS = [
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2)
]

def tf_augment(image, label):
    """Apply pre-created augmentation layers"""
    for layer in AUGMENTATION_LAYERS:
        image = layer(image)
    return image, label

def create_dataloader(images, labels, batch_size=32, shuffle=True):  # Reduced batch size
    # Convert to float32 and normalize here instead of holding full array in memory
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    if shuffle:
        dataset = dataset.shuffle(1024)  # Reduced buffer size
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (tf.image.resize(x, [224, 224]), y),
                         num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_optimized_dataset(images, labels, batch_size=16):
    with cpu_device:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float16) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        if len(images) > 1000:  # Use cache for smaller datasets
            dataset = dataset.cache()
        
        return dataset.batch(batch_size).prefetch(1)

# ========== CORE ARCHITECTURE: RESIDUAL SE-NET ==========
def channel_attention(input_tensor, reduction=8):
    channels = input_tensor.shape[-1]
    x = GlobalAvgPool2D()(input_tensor)
    x = Dense(channels//reduction, activation='swish')(x)
    x = Dense(channels, activation='sigmoid')(x)
    return Multiply()([input_tensor, x])

def residual_block(x, filters, kernel_size=3, stride=1, se_ratio=4):
    shortcut = x
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
    
    # Depthwise Separable Conv
    x = DepthwiseConv2D(kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    
    # Squeeze-and-Excite
    x = channel_attention(x, reduction=se_ratio)
    
    # Pointwise Conv
    x = Conv2D(filters, 1, padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, shortcut])
    return x

# Reduce image size and adjust model parameters
IMG_SIZE = (128, 128)  # Reduced from 224x224
BATCH_SIZE = 16  # Reduced batch size
INITIAL_FILTERS = 16  # Reduced from 32

class GradientCheckpoint(tf.keras.layers.Layer):
    def call(self, x):
        return tf.identity(x)

def create_monster_model(input_shape=(128, 128, 3), num_classes=9):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial Stem with reduced filters
    x = Conv2D(INITIAL_FILTERS, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Reduced number of filters
    filters = [32, 64, 128, 256]
    strides = [1, 2, 2, 2]
    
    # Use layer-based gradient checkpointing
    checkpoint_layer = GradientCheckpoint()
    
    for f, s in zip(filters, strides):
        x = checkpoint_layer(x)  # Use layer instead of function
        x = residual_block(x, f, stride=s)
        x = residual_block(x, f, stride=1)
    
    x = Conv2D(512, 1, activation='relu')(x)  # Reduced from 1024
    x = GlobalAvgPool2D()(x)
    x = Dropout(0.3)(x)  # Reduced dropout
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Use AMP for memory efficiency
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Memory-efficient optimizer
    optimizer = AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4,
        clipnorm=1.0,
        use_ema=True,  # Enable EMA for better memory efficiency
        ema_momentum=0.99
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ========== FUCKING TRAINING LOOP FROM HELL ==========
def train_on_chunk(model, images, labels, validation_data=None):
    with cpu_device:
        # Preprocess on CPU
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        num_classes = len(set(labels_encoded))
        labels_one_hot = to_categorical(labels_encoded, num_classes=num_classes)
        
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels_one_hot, test_size=0.1
        )
        
        train_dataset = create_optimized_dataset(X_train, y_train)
        val_dataset = create_optimized_dataset(X_val, y_val)
    
    with gpu_device:
        # Train on GPU
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=2,  # Reduced epochs per chunk
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=3,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
    
    return model

def main():
    # Initialize progressive loader
    loader = ProgressiveLoader("./Cropped_Components", chunk_size=500)
    
    # Load first chunk to determine number of classes
    images, labels = loader.load_chunk()
    if images is None:
        raise ValueError("No data found in the dataset")
    
    # Get unique classes
    unique_classes = len(set(labels))
    print(f"Found {unique_classes} classes in the dataset")
    
    # Create model with correct number of classes
    model = create_monster_model(input_shape=(128, 128, 3), num_classes=unique_classes)
    
    # Train on first chunk
    model = train_on_chunk(model, images, labels)
    del images, labels
    
    # Progressive training on remaining chunks
    while True:
        tf.keras.backend.clear_session()
        images, labels = loader.load_chunk()
        if images is None:
            break
        model = train_on_chunk(model, images, labels)
        del images, labels
    
    model.save('final_model.keras')

if __name__ == "__main__":
    main()