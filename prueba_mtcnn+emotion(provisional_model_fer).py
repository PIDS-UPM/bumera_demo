import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import time
from fer import FER

def expand_bounding_box(x, y, width, height, expansion_factor, frame_width, frame_height):
    delta_w = int(width * (expansion_factor - 1) / 2)
    delta_h = int(height * (expansion_factor - 1) / 2)
    return (max(0, x - delta_w),
            max(0, y - delta_h),
            min(width + 2 * delta_w, frame_width - (x - delta_w)),
            min(height + 2 * delta_h, frame_height - (y - delta_h)))

def init_detectors():
    print("Inicializando detectores...")
    return MTCNN(), FER(mtcnn=False)

def init_camera():
    print("Abriendo cámara con DirectShow...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(2)
    if not cap.isOpened():
        raise Exception("Error: No se pudo abrir la cámara")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def draw_face_info(frame, x, y, width, height, confidence, box_color, dominant_emotion, emotion_scores, EMOTION_ORDER, EMOTION_COLORS):
    # Dibujar rectángulo y confianza MTCNN
    cv2.rectangle(frame, (x, y), (x + width, y + height), box_color, 2)
    cv2.putText(frame, f'Face conf: {confidence:.2f}', (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    # Mostrar emoción predominante
    cv2.putText(frame, f'Emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})', 
                (x, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EMOTION_COLORS[dominant_emotion[0]], 2)
    
    # Mostrar todas las emociones
    y_offset = y - 65
    for emotion in EMOTION_ORDER:
        cv2.putText(frame, f'{emotion}: {emotion_scores[emotion]:.2f}', 
                    (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, EMOTION_COLORS[emotion], 1)
        y_offset -= 15

def process_face(frame, face_idx, face, emotion_detector, EXPANSION_FACTOR, EMOTION_COLORS, EMOTION_ORDER):
    x, y, width, height = face['box']
    confidence = face['confidence']
    frame_height, frame_width = frame.shape[:2]
    
    # Expandir y extraer ROI
    roi_coords = expand_bounding_box(x, y, width, height, EXPANSION_FACTOR, frame_width, frame_height)
    face_roi = frame[roi_coords[1]:roi_coords[1]+roi_coords[3], roi_coords[0]:roi_coords[0]+roi_coords[2]]
    
    if face_roi.size > 0:
        emotions = emotion_detector.detect_emotions(face_roi)
        if emotions and len(emotions) > 0:
            emotion_scores = emotions[0]['emotions']
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            box_color = EMOTION_COLORS.get(dominant_emotion[0], (0, 255, 0))
            
            draw_face_info(frame, x, y, width, height, confidence, box_color, 
                          dominant_emotion, emotion_scores, EMOTION_ORDER, EMOTION_COLORS)
            
            # Dibujar landmarks
            for point in face['keypoints'].values():
                cv2.circle(frame, point, 2, (0, 0, 255), 2)
                
            # Imprimir datos
            current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"{current_timestamp},face_{face_idx + 1},{dominant_emotion[0]},{dominant_emotion[1]:.2f}")

def main():
    # Configuraciones
    EXPANSION_FACTOR = 1.5
    EMOTION_COLORS = {
        'angry': (0, 0, 255), 'disgust': (0, 140, 255), 'fear': (0, 255, 255),
        'happy': (0, 255, 0), 'sad': (255, 0, 0), 'surprise': (255, 0, 255),
        'neutral': (255, 255, 255)
    }
    EMOTION_ORDER = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    detector, emotion_detector = init_detectors()
    cap = init_camera()
    prev_time = time.time()
    
    print("Iniciando detección de rostros y emociones. Presiona 'q' para salir.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Error al leer el frame")
            
            # Calcular FPS
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            
            # Detectar rostros
            faces = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Procesar cada rostro
            for face_idx, face in enumerate(faces):
                process_face(frame, face_idx, face, emotion_detector, 
                           EXPANSION_FACTOR, EMOTION_COLORS, EMOTION_ORDER)
            
            # Mostrar información general
            cv2.putText(frame, f'Rostros detectados: {len(faces)}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('MTCNN Face Detection + Emotion', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados")

if __name__ == "__main__":
    main()