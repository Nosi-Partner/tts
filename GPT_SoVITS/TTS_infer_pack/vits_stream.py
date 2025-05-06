import torch
import numpy as np
from TTS_infer_pack.TTS_config import TTS_Config
from module.models import SynthesizerTrn


def find_split_point(audio: torch.Tensor, window: int = 12800) -> int:
    """Find a suitable split point in the audio by looking for zero crossings"""
    end = len(audio) - 1
    start = max(0, end - window)
    
    # Try to find quiet zero crossings first
    thresholds = [0.002, 0.005, 0.01]
    for threshold in thresholds:
        for i in range(end, start, -1):
            if ((audio[i] >= 0 and audio[i-1] < 0) or 
                (audio[i] < 0 and audio[i-1] >= 0)) and \
                abs(audio[i]) < threshold and abs(audio[i-1]) < threshold:
                return i-1
    
    # Fall back to any zero crossing if no quiet one found
    for i in range(end, start, -1):
        if (audio[i] >= 0 and audio[i-1] < 0) or (audio[i] < 0 and audio[i-1] >= 0):
            return i-1
    
    return len(audio) - 1

class VITSDecoder:
    def __init__(self, vits_model):
        self.vits_model = vits_model
        self._cancelled = False
        
    def decode_window(self, semantic_window, phones, refer_spec, speed_factor, 
                     is_first_window, is_last_window):
        """Decode a semantic window and return an audio chunk"""
        first_chunk_size = len(semantic_window[0].squeeze())
        semantic_window = torch.cat(semantic_window, dim=1)[0, :]
        
        # Generate audio from semantic input
        audio_window = self.vits_model.decode(
            semantic_window.unsqueeze(0).unsqueeze(0), 
            phones, 
            refer_spec, 
            speed=speed_factor
        ).detach()[0, 0, :]

        # Calculate indices for trimming
        semantic_start_idx = 0 if is_first_window else first_chunk_size
        semantic_end_idx = len(semantic_window) if is_last_window else semantic_start_idx + first_chunk_size

        audio_start_idx = int(semantic_start_idx * 1280 * speed_factor)
        audio_end_idx = int(semantic_end_idx * 1280 * speed_factor)

        # Handle overlap for non-first windows
        if not is_first_window:
            prefix_audio_window = audio_window[:audio_start_idx]
            audio_start_idx -= len(prefix_audio_window) - find_split_point(prefix_audio_window)

        # Extract and process the audio chunk
        audio_chunk = audio_window[audio_start_idx:audio_end_idx]
        audio_chunk_out = audio_chunk if is_last_window else audio_chunk[:find_split_point(audio_chunk)]

        # Normalize if needed
        max_audio = torch.abs(audio_chunk_out).max()
        if max_audio > 1:
            audio_chunk_out /= max_audio
            
        return audio_chunk_out.cpu().numpy().astype(np.float32)

    def stream_audio(self, semantic_stream, phones, refer_spec, speed_factor):
        """Stream audio from semantic input"""
        self._cancelled = False
        semantic_window = []
        window_idx = 0
        
        for chunk_data in semantic_stream:
            if self._cancelled:
                break
                
            chunk, is_last_window = chunk_data
            
            # Maintain sliding window of semantic chunks
            if len(semantic_window) == 3:
                semantic_window = semantic_window[1:]
            semantic_window.append(chunk)
            
            # Process window when we have enough chunks or at the end
            if is_last_window or len(semantic_window) >= 2:
                audio_chunk = self.decode_window(
                    semantic_window.copy(),
                    phones,
                    refer_spec,
                    speed_factor,
                    window_idx == 0,
                    is_last_window
                )
                yield audio_chunk
                window_idx += 1
                
            if is_last_window:
                break

    def cancel(self):
        """Cancel the current stream processing"""
        self._cancelled = True