# signlanguage_aid_translater
 A Sign Language Translator application that uses computer vision and deep learning to detect and translate hand gestures in real-time. The project combines MediaPipe's hand tracking technology with a custom CNN model trained on ASL (American Sign Language) gestures.


# Sign Language Translator

A real-time sign language translation system using computer vision and deep learning to detect and interpret ASL (American Sign Language) gestures.



## Features

- **Real-time Sign Detection**: Recognizes 6 common ASL gestures: Hello, Thank You, Yes, No, Perfect, and Bye
- **Modern UI**: Clean, responsive interface with dark theme
- **Text-to-Speech**: Speaks detected signs aloud
- **Confidence Meter**: Visual representation of prediction confidence
- **Detection History**: Tracks and displays previously detected signs
- **Performance Monitoring**: Real-time FPS counter for performance tracking

## Requirements

- Python 3.10 (recommended for MediaPipe compatibility)
- Webcam or camera device
- Libraries: TensorFlow, MediaPipe, OpenCV, pyttsx3, ttkbootstrap

## Installation

1. Clone the repository:
   
2. Create and activate a virtual environment:
python -m venv .venv(ON VS CODE, PYTHON 10.9.9)

in your termibal write : .venv\Scripts\activate


3. Install dependencies

 Train the model (optional, pre-trained model included)


4. Use the UI controls:
- **Start Camera**: Initialize webcam and gesture recognition
- **Stop Camera**: Stop webcam feed
- **Speak**: Audibly speak the currently detected gesture
- **Exit**: Close the application

## Project Structure

- `sign.py`: Main application with GUI and gesture recognition logic
- `model.py`: Model training script and dataset processing
- `dataset/`: Directory containing gesture image data
- `gesture_model.h5`: Pre-trained gesture recognition model
- `class_names.npy`: Saved class names for gesture mapping

## Model Architecture

The gesture recognition model uses a CNN architecture:
- 3 Convolutional layers with BatchNormalization
- GlobalAveragePooling followed by Dense layers
- Dropout and L2 regularization to prevent overfitting
- Trained with data augmentation and learning rate scheduling

## How It Works

1. **Hand Detection**: MediaPipe identifies hand landmarks in the webcam feed
2. **Feature Extraction**: Hand region is isolated and processed for the model
3. **Gesture Classification**: CNN model predicts the gesture
4. **Result Stabilization**: Predictions are stabilized using a buffer
5. **Feedback**: UI displays the detected gesture and confidence level

## License

[MIT License](LICENSE)

## Acknowledgments

- Hand Gestures Dataset: Used for training the gesture recognition model
- MediaPipe: For hand landmark detection
- TensorFlow: For model training and inference
- ttkbootstrap: For the modern UI elements
 




