import cv2  
import os  
import numpy as np  
import moviepy as mp  
import whisper
from PIL import Image
import uuid
from pathlib import Path 
import requests

class VideoProcessor:  
    def __init__(self):  
        self.temp_audio_path_dir = "temp/"
        self.temp_image_path = "temp/video_frames"
        os.makedirs("temp", exist_ok=True)

    def sample_frames(self, video_path, max_frame_num, save_files=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file at {video_path}")
        if save_files:
            os.makedirs(self.temp_image_path, exist_ok=True)
        sampled_frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_num_frames//fps > max_frame_num:
            interval = (total_num_frames + max_frame_num - 1) // max_frame_num  
        else:
            interval = fps
        current_frame = 0
        saved_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame % interval == 0:
                if save_files:
                    frame_path = os.path.join(self.temp_image_path, f"frame_{saved_frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                sampled_frames.append(frame)
                saved_frame_count += 1

            current_frame += 1

        cap.release()
        sampled_frames = [Image.fromarray(x) for x in sampled_frames]
        return sampled_frames
  
    def extract_audio(self, video_path, sampling_rate=16000):  
        video = mp.VideoFileClip(video_path)  
        audio = video.audio  
        audio_array = audio.to_soundarray(fps=sampling_rate) 
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1) 
        return audio_array, sampling_rate
    
    def get_transcript(self, video_path):  
        remote_url = "http://10.245.95.242:5000/api/transcript"
        server_video_path = "/users/kevin/weijia/videowebarena/" + video_path
       
        payload = {
                    "server_video_path": server_video_path
                }
        print("connecting to whisper_server, video_path: ",server_video_path)
        result = requests.post(remote_url, json=payload)
        result_json = result.json()  # 解析返回的 JSON 数据
        print("successfully got whisper result")
        # print("whisper_result: ",result_json["texts"])
        return result_json["texts"] 
    
    def clean_up(self):
        try:
            os.remove("temp")
        except:
            print("Error removing temp directory")

if __name__ == "__main__":
    video_path = "../media/edit_listings.mov"
    video_processor = VideoProcessor()
    transcript = video_processor.get_transcript(video_path)
    print(transcript)
