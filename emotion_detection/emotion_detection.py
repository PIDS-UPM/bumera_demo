# emotion_detection.py
import cv2
from mtcnn.mtcnn import MTCNN
from fer import FER
import time
from firebase_handler import FirebaseManager

class EmotionDetector:
    """Handles emotion detection"""
    
    def __init__(self, camera_width=1280, camera_height=720, show_roi=True):
        # Initialize detectors and camera
        self.face_detector = MTCNN()
        self.emotion_detector = FER(mtcnn=False)
        self.camera = self._setup_camera(camera_width, camera_height)
        self.show_roi = show_roi  
        
        # Configuration
        self.config = {
            'box_expansion': 1.5,    # Factor to expand face detection box
            'upload_interval': 2.0,  # Seconds between uploads per face
            'colors': {              # BGR colors for each emotion
                'angry': (0, 0, 255),
                'disgust': (0, 140, 255),
                'fear': (0, 255, 255),
                'happy': (0, 255, 0),
                'sad': (255, 0, 0),
                'surprise': (255, 0, 255),
                'neutral': (255, 255, 255)
            }
        }
        
        self.firebase = FirebaseManager()

    def _setup_camera(self, width, height):
        """Initialize and configure the camera"""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        time.sleep(1)  # Allow camera to adjust
        return cap

    def _get_face_roi(self, frame, face):
        """Extract and expand face region of interest for the FER model"""
        x, y, w, h = face['box']
        frame_h, frame_w = frame.shape[:2]
        expand = self.config['box_expansion']
        
        # Calculate expanded box dimensions
        dx = int(w * (expand - 1) / 2)
        dy = int(h * (expand - 1) / 2)
        
        # Ensure coordinates stay within frame
        x1 = max(0, x - dx)
        y1 = max(0, y - dy)
        x2 = min(x + w + dx, frame_w)
        y2 = min(y + h + dy, frame_h)
        
        roi = frame[y1:y2, x1:x2]
        expanded_box = (x1, y1, x2-x1, y2-y1) if self.show_roi else None
        
        return roi, face['box'], expanded_box

    def _draw_dashed_line(self, frame, pt1, pt2, color, thickness, dash_length):
        """Helper function to draw dashed lines"""
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        dashes = int(dist / dash_length)
        for i in range(dashes):
            start_ratio = i / dashes
            end_ratio = (i + 0.5) / dashes
            start = (int(pt1[0] * (1-start_ratio) + pt2[0] * start_ratio),
                    int(pt1[1] * (1-start_ratio) + pt2[1] * start_ratio))
            end = (int(pt1[0] * (1-end_ratio) + pt2[0] * end_ratio),
                  int(pt1[1] * (1-end_ratio) + pt2[1] * end_ratio))
            cv2.line(frame, start, end, color, thickness)

    def _draw_info(self, frame, face, orig_box, expanded_box, emotion_data, face_id):
        """Draw detection results on frame"""
        x, y, w, h = orig_box
        color = self.config['colors'][emotion_data['emotion']]
        
        # Draw original detection box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw ROI box if enabled
        if self.show_roi and expanded_box:
            ex, ey, ew, eh = expanded_box
            roi_color = (0, 255, 255)  # Yellow color for ROI
            
            # Draw expanded box with dashed lines
            points = [(ex, ey), (ex+ew, ey),
                     (ex+ew, ey+eh), (ex, ey+eh)]
            for i in range(4):
                self._draw_dashed_line(frame, points[i], points[(i+1)%4],
                                     roi_color, 1, 10)
            
            # Add ROI label
            cv2.putText(frame, 'ROI', (ex, ey-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)
        
        # Draw face information
        info_text = [
            f'Student ID: {self.firebase.student_mapping.get(face_id, "Unknown")}',
            f'Confidence: {face["confidence"]:.2f}',
            f'Emotion: {emotion_data["emotion"]} ({emotion_data["confidence"]:.2f})'
        ]
        
        for i, text in enumerate(info_text):
            y_pos = y - 10 - (20 * i)
            cv2.putText(frame, text, (x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255) if i < 2 else color, 2)

    def process_frame(self, frame):
        """Process a single frame for face detection and emotion analysis"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.detect_faces(rgb_frame)
        
        for i, face in enumerate(faces):
            face_id = f"face_{i + 1}"
            
            # Get face region and detect emotion
            roi, orig_box, expanded_box = self._get_face_roi(frame, face)
            if roi.size == 0:
                continue
                
            emotions = self.emotion_detector.detect_emotions(roi)
            if not emotions:
                continue
            
            # Get dominant emotion
            emotion_scores = emotions[0]['emotions']
            emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
            
            # Draw results and upload data
            self._draw_info(frame, face, orig_box, expanded_box,
                          {'emotion': emotion, 'confidence': confidence},
                          face_id)
            
            if self.firebase.should_upload(face_id):
                self.firebase.upload_emotion_data(
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    face_id,
                    emotion,
                    confidence
                )
        
        return len(faces)

    def run(self):
        """Main loop for emotion detection"""
        print("Starting emotion detection. Press 'q' to quit.")
        last_time = time.time()
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                fps = 1 / (time.time() - last_time)
                last_time = time.time()
                
                face_count = self.process_frame(frame)
                
                cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Faces: {face_count}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                cv2.imshow('Emotion Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set show_roi=False to disable ROI visualization
    detector = EmotionDetector(show_roi=False)
    detector.run()
    