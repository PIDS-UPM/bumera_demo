#____________________________________________________________________


###### PRUEBA FALLIDA 1 #####



# from modelscope.utils.constant import Tasks
# from modelscope.pipelines import pipeline
# from modelscope.hub.snapshot_download import snapshot_download
# from modelscope.hub.api import HubApi
# import json

# def explore_available_models():
#     print("Exploring available speech separation models...\n")
    
#     try:
#         # Try to initialize a pipeline without specifying model
#         print("Available tasks:")
#         print(Tasks.list_tasks())
        
#         print("\nTrying to get information about Mossformer2...")
#         try:
#             # Try to download model information
#             model_id = 'damo/speech_mossformer2_separation_temporal_8k'
#             info = snapshot_download(model_id, revision='master')
#             print(f"Model info: {info}")
#         except Exception as e:
#             print(f"Error with Mossformer2: {str(e)}")
        
#         print("\nTrying alternative models:")
#         alternative_models = [
#             'damo/speech_separation_nsnet2_8k',
#             'damo/speech_mossformer2_separation_temporal_8k',
#             'damo/speech_separation_mossformer_8k'
#         ]
        
#         for model_id in alternative_models:
#             try:
#                 print(f"\nTrying model: {model_id}")
#                 test_pipeline = pipeline(Tasks.speech_separation, model=model_id)
#                 print(f"Successfully loaded: {model_id}")
#             except Exception as e:
#                 print(f"Failed to load {model_id}: {str(e)}")
                
#     except Exception as e:
#         print(f"General error: {str(e)}")

# if __name__ == "__main__":
#     explore_available_models()
    

#____________________________________________________________________



###### PRUEBA FALLIDA 2 #####



# import torch
# import soundfile as sf
# from transformers import AutoModel

# # Load model
# model = AutoModel.from_pretrained("alibabasglab/mossformer2-wsj0mix-3spk")
# model.eval()

# # Use GPU if available
# if torch.cuda.is_available():
#     model = model.cuda()

# # Load and process audio
# input_file = 'audiopruebausar.wav'
# audio_data, sr = sf.read(input_file)
# audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)

# if torch.cuda.is_available():
#     audio_tensor = audio_tensor.cuda()

# # Separate audio
# with torch.no_grad():
#     separated_signals = model(audio_tensor)

# # Save separated audio files
# for i, signal in enumerate(separated_signals):
#     signal = signal.cpu().numpy()
#     sf.write(f'separated_speaker_{i+1}.wav', signal.squeeze(), sr)

#____________________________________________________________________



###### PRUEBA FALLIDA 3 #####

import torch
import soundfile as sf
import torchaudio
from torch import nn
import os
from transformers import AutoConfig
import requests
from huggingface_hub import hf_hub_download

class AudioSeparator:
    def __init__(self, model_path="alibabasglab/mossformer2-wsj0mix-3spk"):
        # First check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load custom model using torch hub instead of transformers
        try:
            self.model = torch.hub.load('huggingface/pytorch-transformers', 
                                      'model', 
                                      model_path,
                                      force_reload=True)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting alternative loading method...")
            # Alternative: Download model files directly
            try:
                model_file = hf_hub_download(repo_id=model_path, 
                                           filename="pytorch_model.bin")
                config_file = hf_hub_download(repo_id=model_path, 
                                            filename="config.json")
                self.model = torch.load(model_file, map_location=self.device)
            except Exception as e:
                print(f"Alternative loading also failed: {e}")
                raise

    def load_audio(self, file_path):
        """Load and preprocess audio file"""
        try:
            audio_data, sr = sf.read(file_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono if stereo
            audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
            return audio_tensor.to(self.device), sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            raise

    def separate(self, audio_tensor):
        """Separate audio sources"""
        with torch.no_grad():
            try:
                separated_signals = self.model(audio_tensor)
                return separated_signals
            except Exception as e:
                print(f"Error during separation: {e}")
                raise

    def save_separated_audio(self, separated_signals, sr, output_dir="separated_outputs"):
        """Save separated audio files"""
        os.makedirs(output_dir, exist_ok=True)
        try:
            for i, signal in enumerate(separated_signals):
                signal = signal.cpu().numpy()
                output_path = os.path.join(output_dir, f'separated_speaker_{i+1}.wav')
                sf.write(output_path, signal.squeeze(), sr)
                print(f"Saved separated audio to: {output_path}")
        except Exception as e:
            print(f"Error saving separated audio: {e}")
            raise

def main():
    try:
        # Initialize separator
        separator = AudioSeparator()
        
        # Load audio
        input_file = 'audiopruebausar.wav'
        audio_tensor, sr = separator.load_audio(input_file)
        
        # Separate audio
        separated_signals = separator.separate(audio_tensor)
        
        # Save results
        separator.save_separated_audio(separated_signals, sr)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()