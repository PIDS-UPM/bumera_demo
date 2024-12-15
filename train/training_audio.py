import librosa
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import joblib

#Leectura de audios del dataset SAVEE
audio_dir = 'SAVEE_dataset/ALL'
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

# parse the filename to get the emotions
emotion=[]
emotion_map = {
    '_a': 'angry',
    '_d': 'disgust',
    '_f': 'fear',
    '_h': 'happy',
    '_n': 'neutral',
    'sa': 'sad',
    'su': 'surprise'
}


def extract_descriptors(y, sr):
    
    #1. MFCC (Mel Frequency Cepstral Coefficient)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)

    #2. Chroma feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    #3.Pitch
    pitch, mag = librosa.core.piptrack(y=y, sr=sr)
    pitch_values = pitch[pitch>0]
    pitch_mean = np.mean(pitch_values) if pitch_values.size > 0 else 0

    #4.Ring using spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    #5.Zero-Crossing Rate
    zero_crossing = librosa.feature.zero_crossing_rate(y=y)

    #6. Root Mean Square Error Energy
    rms = librosa.feature.rms(y=y)

    # Combine features in only one vector
    feature_vector = np.hstack((
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        [pitch_mean],
        np.mean(spectral_centroid, axis=1),
        np.mean(zero_crossing, axis=1),
        np.mean(rms)
    ))

    return feature_vector

# Data augmentation fuction
def augment_audio(y, sr, augmentation_type):
    if augmentation_type == "pitch_shift":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=2, bins_per_octave=12)
    elif augmentation_type == "time_stretch":
        return librosa.effects.time_stretch(y, rate=1.2)
    elif augmentation_type == "add_noise":
        noise = np.random.normal(0, 0.02, len(y))
        return y + noise
    return y

data = []

#Sotre for each audio its descriptors and labels
for audio_file in audio_files:
    for key, emotion in emotion_map.items():
        if key in audio_file:
            audio_path = os.path.join(audio_dir, audio_file)
            y,sr = librosa.load(audio_path)
            descriptors = extract_descriptors(y,sr)
            data.append([audio_file, emotion] + list(descriptors))
            
            for aug_type in ['pitch_shift', 'time_stretch', 'add_noise']:
                y_augmented = augment_audio(y, sr, aug_type)
                augmented_descriptors = extract_descriptors(y_augmented, sr)
                data.append([f"{audio_file.split('.')[0]}_{aug_type}", emotion] + list(augmented_descriptors))
            break



#Create the SAVEE dataset
columns = ['Audio', 'Emotion'] + [f'Descriptor_{i}' for i in range(1, 1 + len(descriptors))]
SAVEE_df_augmented = pd.DataFrame(data, columns=columns)

print(SAVEE_df_augmented)

# SAVEE DATASET STUCTURE:
# Audio -> name of the audio
# Emotion -> Label asociated
# Descriptors:
#      - 1 to 30 -> MFCCS
#      - 31 to 43 -> Chroma
#      - 44 -> Pitch
#      - 45 -> SC (Spectral Centroid)
#      - 46 -> ZCR (Zero Crossing Rate)
#      - 47 -> RSM (Root Mean Squared Error)

# Modelado de ML con varios modelos para detección de emociones
X = SAVEE_df_augmented.drop(columns=['Audio','Emotion'], inplace=False) #Characteristics
y = SAVEE_df_augmented['Emotion'] #Labels

#Divide data in test and training 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state=42, stratify=y)

#Normalize characteristics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.fit_transform(X_test)

#Models to evaluate performance
models = {
    'Neural Network (MLP)': MLPClassifier(max_iter=3000,random_state=42)
}

param_grids = {
    'Neural Network (MLP)': {
        'hidden_layer_sizes': [(30,), (10,10), (15,15), (30,30), (45,45,15)],
        'activation': ['relu', 'tanh'],
        'solver': ['sgd'],
        'learning_rate':['adaptive'],
        'learning_rate_init' :[0.01, 0.05, 0.1],
        'alpha': [0.0001, 0.001, 0.01]
    }
}


with open('audio_results2.txt', 'w') as f: # File to store the best results and hyperparams
    for model_name, model in models.items():

        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        
        # Training using GridSearch
        grid_search.fit(X_train_scaled, y_train)
        
        # Select best hyperparams combination
        best_params = grid_search.best_params_
        f.write(f"Best Hyperparameters for {model_name}: {best_params}\n")
        
        # Select best model
        best_model = grid_search.best_estimator_

        # Save the trained model to a file
        model_filename = f"{model_name}_best_model.pkl"
        joblib.dump(best_model, model_filename)  
        f.write(f"Model saved as: {model_filename}\n")
        
        #Evaluate with best model
        start_predict = time.time()
        y_pred = best_model.predict(X_test_scaled)
        end_predict = time.time()
        
        accuracy = accuracy_score(y_test, y_pred)
        f.write(f'{model_name} - Accuracy with Best Params: {accuracy:.2f}\n')
        f.write(f'{model_name} - Prediction Time: {end_predict - start_predict:.2f} seconds\n')

        if model_name in ['Gradient Boosting']:
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
            disp.plot(cmap='viridis')
            plt.title(f'Confusion Matrix for {model_name}')
            output_path = f'{model_name}_confusion_matrix2.png'  
            plt.savefig(output_path)  
            print(f"Confusion Matrix saved as {output_path}")  
            plt.close()
        
        report = classification_report(y_test, y_pred)
        f.write("Classification Report:\n")
        f.write(report + "\n")
        
        f.write("\n" + "="*50 + "\n\n")




