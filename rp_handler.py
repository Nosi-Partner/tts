import runpod
import base64
import numpy as np
import sys
import os

now_dir = os.getcwd()
sys.path.insert(0, now_dir)

from rp_engine import TTSEngine

tts_engine = TTSEngine()

ref_text = "The quick brown fox jumps over the lazy dog."
ref_audio_path = "reference.wav"

def handler(event):
    input = event['input']
    text = input.get('text')
    
    for original_text, sample_rate, audio_data in tts_engine.synthesize(
        text=text,
        text_lang="en",
        ref_audio_path=ref_audio_path,
        prompt_text=ref_text,
    ):
        audio_bytes = audio_data.tobytes()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        
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