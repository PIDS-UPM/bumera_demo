# firebase_handler.py
import firebase_admin
from firebase_admin import credentials, firestore
import time
import json
import atexit

class FirebaseManager:
    """Handles data management and Firebase integration"""
    
    def __init__(self, cred_file="key.json", temp_file="emotion_data_buffer.json"):
        # Initialize Firebase
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_file)
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        
        # Setup data storage
        self.temp_file = temp_file
        self.last_upload = {}
        self.data_buffer = {"emotions": []}
        self._load_cached_data()
        
        # Load student mapping from Firebase
        self.student_mapping = self._load_student_mapping()
        
        # Register cleanup handler
        atexit.register(self._cleanup)

    def _load_student_mapping(self):
        """Load student IDs from Firestore students collection"""
        try:
            # Get all documents from students collection
            students_ref = self.db.collection('students')
            students = students_ref.get()
            
            # Create mapping starting from face_1
            mapping = {}
            for i, student in enumerate(students, 1):
                mapping[f"face_{i}"] = student.id
                
            print(f"Loaded {len(mapping)} student IDs from Firebase")
            return mapping
            
        except Exception as e:
            print(f"Error loading student mapping from Firebase: {e}")
            print("Using empty student mapping")
            return {}

    def _load_cached_data(self):
        """Load previously cached data"""
        try:
            with open(self.temp_file, 'r') as f:
                self.data_buffer = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def should_upload(self, face_id, interval=2.0):
        """Check if enough time has passed to upload new data"""
        current_time = time.time()
        if current_time - self.last_upload.get(face_id, 0) >= interval:
            self.last_upload[face_id] = current_time
            return True
        return False

    def upload_emotion_data(self, timestamp, face_id, emotion, confidence):
        """Store emotion data in buffer"""
        if student_id := self.student_mapping.get(face_id):
            self.data_buffer["emotions"].append({
                "student_id": student_id,
                "emotion": emotion,
                "context": "general",
                "timestamp": timestamp
            })
            
            # Periodically save buffer
            if len(self.data_buffer["emotions"]) % 100 == 0:
                self._save_buffer()

    def _save_buffer(self):
        """Save data buffer to temporary file"""
        with open(self.temp_file, 'w') as f:
            json.dump(self.data_buffer, f, indent=2)

    def _cleanup(self):
        """Save and upload all remaining data"""
        print("\nSaving and uploading final data...")
        self._save_buffer()
        
        try:
            # Upload all buffered data
            for i, data in enumerate(self.data_buffer["emotions"]):
                self.db.collection("emotions").document(f"emotions_{i+1}").set(data)
            print("Data upload to Firebase completed")
            
            # Clear buffer
            self.data_buffer = {"emotions": []}
            self._save_buffer()
            
        except Exception as e:
            print(f"Upload failed: {e}")
            print("Data saved locally in buffer for next attempt")

    def reload_student_mapping(self):
        """Manually reload student mapping from Firebase"""
        self.student_mapping = self._load_student_mapping()