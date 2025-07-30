Voice Emotion Recognition using Classical Machine Learning
This project classifies human emotions (like happy, sad, angry, calm) based on voice using classical Machine Learning (SVM). It includes training on .wav audio files, testing, live voice recording for prediction, and result visualization with charts.

✅ Features
Extracts MFCC features from .wav files using librosa
Uses StandardScaler and trains an SVM classifier
Supports live voice input for real-time emotion prediction
Displays predictions using pie charts
Saves and loads the trained model using joblib
📁 Folder & Dataset Structure
Project Folder/
├── train.py                # Main script for training and prediction
├── Sound recordings/       # Folder containing labeled .wav files
│   ├── happy_1.wav
│   ├── sad_2.wav
│   ├── calm_3.wav
│   └── angry_4.wav
Ensure file names start with the emotion label (e.g., happy_1.wav, sad_2.wav).

💻 Requirements
Install the required libraries with:

pip install numpy librosa scikit-learn matplotlib sounddevice soundfile joblib
🚀 How to Run
python train.py
This will:

Extract features from the dataset
Train an SVM model
Save the model to svmodel.pkl
Record live voice twice
Predict emotion from live input
Show a pie chart of results
📊 Example Output
✅ Processed: happy_1.wav -> happy
✅ Processed: sad_2.wav -> sad
🎯 Model Accuracy: 91.00%
🎤 Speak now... (Recording 1)
🧠 Predicted Emotion: HAPPY
🎤 Speak now... (Recording 2)
🧠 Predicted Emotion: SAD
🤖 Technologies Used
Python
Librosa – audio feature extraction
scikit-learn – SVM, pipeline, accuracy
Matplotlib – visualization
sounddevice, soundfile – for live voice recording
👩‍💻 Author
Nimra Fatima AI Researcher | Python & ML Developer
🔗 GitHub: [https://github.com/Nimra71?tab=repositories] | LinkedIn: [https://www.linkedin.com/in/nimra-fatima-b67760333?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app]
