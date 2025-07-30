import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")

# 🎯 1. SETUP: Dataset and emotions
dataset_path = r"C:\\Users\\user\\Documents\\Sound recordings"
emotions = ['happy', 'sad', 'angry', 'calm', 'surprised']


for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            print(f"Found file: {file}")  # Debug line
            # existing code continues...

# 🎧 2. Feature Extraction
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"❌ Error extracting features from {file_path}: {e}")
        return None



# 📥 3. Load Data
X = []
y = []

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            try:
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                if features is not None:
                    # Extract label from filename like 'happy_1.wav'
                    label = file.split("_")[0].lower()
                    X.append(features)
                    y.append(label)
                    print(f"✅ Processed: {file} -> {label}")
                else:
                    print(f"❌ Feature extraction failed for {file}")
            except Exception as e:
                print(f"⚠️ Error processing {file}: {e}")


# 📊 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🧠 5. Train Model
model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
model.fit(X_train, y_train)

# 💾 Save model
dump(model, "emotion_model.joblib")

# 🧪 6. Evaluate
y_pred = model.predict(X_test)
print("\n🎯 Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 🎙️ 7. Live Prediction and Visualization
def record_voice(filename, duration=3, fs=44100):
    print("🎤 Recording... Speak Now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, recording, fs)
    print("✅ Recording finished.")

def predict_emotion(audio_file):
    features = extract_features(audio_file).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction


#  9. Plot results
from collections import Counter
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

# Define recording settings
duration = 4  # seconds
sr = 22050    # sample rate

live_predictions = []

# Record and predict 5 times
for i in range(2):
    print(f"🎤 Recording {i+1}/2... Speak Now!")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    temp_path = "temp_live.wav"
    sf.write(temp_path, recording, sr)
    
    features = extract_features(temp_path).reshape(1, -1)
    # features = scaler.transform(features)
    prediction = model.predict(features)[0]
    
    print(f"🧠 Predicted Emotion: **{prediction.upper()}**")
    live_predictions.append(prediction)

# Count and plot pie chart
emotion_counts = Counter(live_predictions)

plt.figure(figsize=(7, 7))
plt.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', startangle=140)
plt.title("🎤 Live Voice Emotion Predictions")
plt.axis('equal')
plt.show()


