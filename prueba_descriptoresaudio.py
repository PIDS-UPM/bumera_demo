import librosa
import numpy as np
import json

def extraer_caracteristicas_agresion(ruta_audio):
    # Cargar audio
    audio, sr = librosa.load(ruta_audio, sr=None)
    
    # 1. Intensidad (volumen)
    rms = librosa.feature.rms(y=audio)[0]
    intensidad = {
        'media': float(np.mean(rms)),
        'max': float(np.max(rms)),      # Picos de volumen
        'std': float(np.std(rms))       # Variabilidad en volumen
    }
    
    # 2. Tono (F0)
    f0, voiced_flag, _ = librosa.pyin(audio, 
                                     fmin=librosa.note_to_hz('C2'),
                                     fmax=librosa.note_to_hz('C7'))
    f0_valid = f0[~np.isnan(f0)]
    
    tono = {
        'media': float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0,
        'std': float(np.std(f0_valid)) if len(f0_valid) > 0 else 0,
        'proporcion_voz': float(np.mean(voiced_flag)) * 100
    }
    
    # 3. Ritmo y velocidad del habla
    # Cruces por cero (aproximación de velocidad del habla)
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    
    # Detección de onset para patrones rítmicos
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    
    ritmo = {
        'velocidad': float(np.mean(zcr)),           # Velocidad general del habla
        'variabilidad_velocidad': float(np.std(zcr)), # Cambios en velocidad
        'intensidad_onset': float(np.mean(onset_env)) # Fuerza de los inicios de sonido
    }
    
    # 4. Energía en frecuencias altas
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    espectro = {
        'brillo': float(np.mean(cent)),
        'variabilidad': float(np.std(cent))
    }
    
    # 5. Cualidad vocal
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=7)
    timbre = {
        'mfccs': np.mean(mfccs, axis=1).tolist()
    }

    # Crear diccionario completo
    caracteristicas = {
        'intensidad': intensidad,
        'tono': tono,
        'ritmo': ritmo,
        'espectro': espectro,
        'timbre': timbre
    }

    # Crear array con todos los valores (para uso posterior)
    valores = [
        intensidad['media'], intensidad['max'], intensidad['std'],
        tono['media'], tono['std'], tono['proporcion_voz'],
        ritmo['velocidad'], ritmo['variabilidad_velocidad'], ritmo['intensidad_onset'],
        espectro['brillo'], espectro['variabilidad']
    ] + timbre['mfccs']
    
    return caracteristicas, valores

if __name__ == "__main__":
    
    # Extraer características
    caract, valores = extraer_caracteristicas_agresion('audios/audioOPUS1speaker.opus')
    
    # Array de descriptores obtenidos 
    print(valores)