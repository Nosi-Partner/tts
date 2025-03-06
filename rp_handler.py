import runpod
import base64
import numpy as np
import io
import wave
from pydub import AudioSegment
from rp_engine import TTSEngine
from time import time as ttime

tts_engine = TTSEngine()

ref_text = "The quick brown fox jumps over the lazy dog."
ref_audio_path = "reference.wav"

# # Warm up the model first
# for sample_rate, audio_data in tts_engine.synthesize(
#     text="Warm up.",
#     text_lang="en",
#     ref_audio_path=ref_audio_path,
#     prompt_text=ref_text,
# ):
#     continue

def convert_audio_to_base64_opus(sample_rate, audio_data, bitrate='32k'):
    """
    Convert numpy audio data to base64-encoded Opus format.
    Requires ffmpeg to be installed.
    
    Args:
        sample_rate: The sample rate of the audio
        audio_data: NumPy array of audio data
        bitrate: Opus encoding bitrate (default: 32k - good for speech)
        
    Returns:
        Base64-encoded Opus data URI
    """
    # First create a WAV in memory
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    # Convert WAV to Opus using pydub
    wav_io.seek(0)
    audio_segment = AudioSegment.from_wav(wav_io)
    
    # Export as Opus to a new BytesIO object
    opus_io = io.BytesIO()
    audio_segment.export(opus_io, format="opus", bitrate=bitrate, codec="libopus")
    
    # Get Opus bytes and encode to base64
    opus_io.seek(0)
    opus_bytes = opus_io.read()
    base64_audio = base64.b64encode(opus_bytes).decode('utf-8')
    
    # Add the data URI prefix for Opus in an Ogg container
    return f"data:audio/ogg;base64,{base64_audio}"

async def handler(event):
    input = event['input']
    text = input.get('text')
    
    t0 = ttime()
    for norm_text, sample_rate, audio_data in tts_engine.synthesize(
        text=text,
        text_lang="ja",
        ref_audio_path=ref_audio_path,
        prompt_text=ref_text,
    ):
        base64_audio = convert_audio_to_base64_opus(sample_rate, audio_data)
        
        result = {
            "text": norm_text,
            "audio": base64_audio
        }
        
        print(f"{ttime()-t0:.3f}s in total for chunk")
        yield result
        t0 = ttime()

if __name__ == '__main__':
    runpod.serverless.start({
        'handler': handler,
        'return_aggregate_stream': True,
    })