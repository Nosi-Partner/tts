import os
import sys
import re
import numpy as np
from typing import List, Tuple, Generator, Optional

# Add paths
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names

class TTSEngine:
    """
    Text-to-Speech Engine that processes text with <laugh /> tags
    and streams audio segments with their corresponding original text.
    """
    
    def __init__(self, config_path: str = "GPT_SoVITS/configs/tts_infer.yaml"):
        self.i18n = I18nAuto()
        
        # Initialize TTS pipeline
        self.tts_config = TTS_Config(config_path)
        self.tts_pipeline = TTS(self.tts_config)

    def synthesize(self, 
                  text: str,
                  text_lang: str,
                  ref_audio_path: str,
                  aux_ref_audio_paths: Optional[List[str]] = None,
                  prompt_text: str = "",
                  prompt_lang: Optional[str] = None,
                  top_k: int = 5,
                  top_p: float = 1,
                  temperature: float = 1,
                  speed_factor: float = 1.0,
                  seed: int = -1,
                  repetition_penalty: float = 1.35,
                  ) -> Generator[Tuple[str, int, np.ndarray], None, None]:
        """
        Synthesize speech from text and yield audio segments with original text.
        
        Args:
            text: Text to synthesize
            text_lang: Language of the text (e.g., "en", "zh")
            ref_audio_path: Path to reference audio file
            aux_ref_audio_paths: List of paths to auxiliary reference audio files
            prompt_text: Prompt text for the reference audio
            prompt_lang: Language of the prompt text
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            temperature: Temperature for sampling
            speed_factor: Speed factor for synthesis
            seed: Random seed for reproducibility
            repetition_penalty: Repetition penalty for T2S model
            
        Yields:
            Tuples of (original_text, sample_rate, audio_data)
        """
        # Validate inputs
        if not text or not text_lang or not ref_audio_path:
            raise ValueError("text, text_lang, and ref_audio_path are required")

        if text_lang.lower() not in self.tts_config.languages:
            raise ValueError(f"text_lang: {text_lang} not supported in version {self.tts_config.version}")

        if prompt_lang and prompt_lang.lower() not in self.tts_config.languages:
            raise ValueError(f"prompt_lang: {prompt_lang} not supported in version {self.tts_config.version}")
        
        request = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang.lower() if prompt_lang else text_lang.lower(),
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "speed_factor": speed_factor,
            "seed": seed,
            "repetition_penalty": repetition_penalty,
            "parallel_infer": True,
            "return_fragment": True,
            "batch_size": 1,
        }
        
        try:
            for spoken_text, (sample_rate, audio_data) in self.tts_pipeline.run(request):
                yield spoken_text, sample_rate, audio_data
                
        except Exception as e:
            print(f"Synthesis failed for text '{text}': {str(e)}")