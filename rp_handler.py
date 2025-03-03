import runpod
import base64
import numpy as np
import io
import wave
from rp_engine import TTSEngine

tts_engine = TTSEngine()

ref_text = "The quick brown fox jumps over the lazy dog."
ref_audio_path = "reference.wav"

# Warm up the model first
tts_engine.synthesize(
    text="Hey this is just a test to warm up.",
    text_lang="en",
    ref_audio_path=ref_audio_path,
    prompt_text=ref_text,
)

def convert_audio_to_base64_wav(sample_rate, audio_data):
    # Create an in-memory WAV file
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 16-bit audio (2 bytes)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Get the WAV bytes and encode to base64
        wav_io.seek(0)
        wav_bytes = wav_io.read()
        base64_audio = base64.b64encode(wav_bytes).decode('utf-8')
    
    # Add the data URI prefix
    return f"data:audio/wav;base64,{base64_audio}"

def handler(event):
    input = event['input']
    text = input.get('text')
    
    for original_text, sample_rate, audio_data in tts_engine.synthesize(
        text=text,
        text_lang="en",
        ref_audio_path=ref_audio_path,
        prompt_text=ref_text,
    ):
        base64_audio = convert_audio_to_base64_wav(sample_rate, audio_data)
        
        result = {
            "text": original_text,
            "sample_rate": sample_rate,
            "audio_data_base64": base64_audio
        }
        
        yield result

if __name__ == '__main__':
    runpod.serverless.start({
        'handler': handler,
        'return_aggregate_stream': True,
    })