# import hydra
import soundfile as sf
import torch
from omegaconf import OmegaConf
import locale
from io import BytesIO, BufferedReader
from typing import List, Tuple

import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment
locale.getpreferredencoding = lambda: "UTF-8"
import pandas as pd
import torch
import torchaudio
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)

# ------------------ Custom Preprocessor Classes ------------------

# Custom class to handle filterbank features with additional options


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=kwargs["nfilt"],
                window_fn=self.torch_windows[kwargs["window"]],
                mel_scale=mel_scale,
                norm=kwargs["mel_norm"],
                n_fft=kwargs["n_fft"],
                f_max=kwargs.get("highfreq", None),
                f_min=kwargs.get("lowfreq", 0),
                wkwargs=wkwargs,
            )
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA( 
                mel_scale=mel_scale,
                **kwargs,
            )
        )



# ------------------ Audio Processing Functions ------------------

# Convert an AudioSegment object to a numpy array

def audiosegment_to_numpy(audiosegment: AudioSegment) -> np.ndarray:
    samples = np.array(audiosegment.get_array_of_samples())
    if audiosegment.channels == 2:
        samples = samples.reshape((-1, 2))

    samples = samples.astype(np.float32, order="C") / 32768.0
    return samples

# Segment the audio file based on speech activity detection (SAD)


def segment_audio(
    audio_path: str,
    pipeline: Pipeline,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    new_chunk_threshold: float = 0.2,
) -> Tuple[List[np.ndarray], List[List[float]]]:
    audio = AudioSegment.from_wav(audio_path)
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    sad_segments = pipeline({"uri": "filename", "audio": audio_bytes})

    segments = []
    curr_duration = 0
    curr_start = 0
    curr_end = 0
    boundaries = []

    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(len(audio) / 1000, segment.end)
        if (
            curr_duration > min_duration and start - curr_end > new_chunk_threshold
        ) or (curr_duration + (end - curr_end) > max_duration):
            audio_segment = audiosegment_to_numpy(
                audio[curr_start * 1000 : curr_end * 1000]
            )
            segments.append(audio_segment)
            boundaries.append([curr_start, curr_end])
            curr_start = start

        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration != 0:
        audio_segment = audiosegment_to_numpy(
            audio[curr_start * 1000 : curr_end * 1000]
        )
        segments.append(audio_segment)
        boundaries.append([curr_start, curr_end])

    return segments, boundaries

# Convert a stereo audio file to mono


def convert_to_mono(audio_input):
    try:
        waveform, sample_rate = torchaudio.load(audio_input)
        if waveform.shape[0] > 1:  
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        temp_audio_path = "temp_mono_audio.wav"
        torchaudio.save(temp_audio_path, waveform, sample_rate)
        return temp_audio_path
    except Exception as e:
        print(f"Failed to process audio file {audio_input}: {e}")
        return None


# ------------------ Model Loading and Inference ------------------

# Load the ASR model from a configuration file and checkpoint

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncDecRNNTBPEModel.from_config_file("/app/config/rnnt_model_config.yaml")
    ckpt = torch.load("/app/rnnt_model_weights.ckpt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model.to(device)

# Process a single audio file and transcribe it using the model


def process_audio_file(audio_path, model):
    mono_audio_path = convert_to_mono(audio_path)
    if mono_audio_path is None:
        return "Audio processing failed"
    try:
        transcription = model.transcribe([mono_audio_path])[0]
        return transcription
    except Exception as e:
        print(f"Failed to transcribe audio file {mono_audio_path}: {e}")
        return "Transcription failed"


# ------------------ Batch Processing ------------------

# Process a CSV file containing paths to audio files and save transcriptions

def process_csv(file_path, model):
    df = pd.read_csv(file_path)
    df['inference_text'] = df['audio_path'].apply(lambda x: process_audio_file(x, model))
    df.to_csv(file_path, index=False)
