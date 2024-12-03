import os
from moviepy import VideoFileClip
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    ViTImageProcessor,
    pipeline,
    T5Tokenizer,
    T5ForConditionalGeneration
)
import torch
import cv2
import uuid

# Global model and processor initialization for performance
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Vision-Encoder-Decoder Model
vision_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
vision_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vision_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Whisper for transcription
transcriber = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-small", 
    device=0 if torch.cuda.is_available() else -1, 
    return_timestamps=True  # Enable long-form transcription
)

# T5 for summarization
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)


def extract_frames(video_path, interval=30):
    """Extract frames from the video at the given interval."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval_frames = interval * fps
    extracted_frames = []

    for i in range(0, frame_count, interval_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            extracted_frames.append(frame)

    cap.release()
    return extracted_frames


def generate_frame_descriptions(frames):
    """Generate descriptions for video frames."""
    descriptions = []
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pixel_values = vision_processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = vision_model.generate(pixel_values)
        caption = vision_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        descriptions.append(caption)

    return descriptions


def extract_transcription(video_path):
    """Extract and transcribe audio using Whisper."""
    audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec="pcm_s16le")

    # Process long-form audio transcription
    transcription_result = transcriber(audio_path)
    transcription = transcription_result["text"]  # Full transcription without truncation
    os.remove(audio_path)  # Clean up temporary file
    return transcription


def summarize_content(content):
    """Summarize content using T5."""
    inputs = t5_tokenizer(
        content,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # Process in chunks of up to 512 tokens
        padding="max_length",
    ).to(device)

    summary_ids = t5_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1024,  # Generate long summaries
        min_length=150,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=False,  # Ensure full content generation
    )
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def summarize_video(video_path):
    """Main function to process video content."""
    try:
        print("Extracting frames from video...")
        frames = extract_frames(video_path)
        print(f"Extracted {len(frames)} frames.")

        print("Generating frame descriptions...")
        frame_descriptions = generate_frame_descriptions(frames)
        video_description = " ".join(frame_descriptions)

        print("Extracting and transcribing audio...")
        transcription = extract_transcription(video_path)

        # Combine video descriptions and audio transcription for summary
        combined_content = (
            f"Video Description: {video_description}\n"
            f"Audio Transcription: {transcription}"
        )
        print("Generating final summary...")
        final_summary = summarize_content(combined_content)

        return {
            "video_description": video_description,
            "audio_transcription": transcription,
            "summary": final_summary
        }

    except Exception as e:
        print(f"Error during video processing: {e}")
        return {"error": "An error occurred during video processing."}
