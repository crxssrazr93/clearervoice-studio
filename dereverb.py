"""
Room echo removal (de-reverberation) for speech.

ClearerVoice's enhancement models remove background noise but leave room reverb
almost untouched (the echo tail after each word drops only 1 to 3 dB). This module
runs anvuew's mono MelBand-RoFormer de-reverb model, which removes about 10 dB of
that tail on real talking head recordings. The worker runs it BEFORE the selected
enhancement model, which then cleans up the remaining noise.

Model: anvuew/dereverb_mel_band_roformer, dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt
Do NOT swap in the sdr_19.1729 or less_aggressive checkpoints from the same repo:
they were trained under an MSST data loading bug (documented in the model README)
and pass reverberant speech through unchanged.

The model was trained on mono, single voice dry speech. It treats any difference
between the channels as reverb, so the input is always folded to mono and fed as
dual mono. Feeding it real stereo wipes out most of the signal.
"""
import hashlib
import os
import time
import urllib.request

import numpy as np

MODEL_FILE = "dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt"
MODEL_URL = ("https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/"
             + MODEL_FILE)
MODEL_SHA256 = "f099ee717eb57fb0ad5eb0e7c9ad6787c36168140b61ce2b158b90c2c4ecee79"

APP = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(APP, "checkpoints", "dereverb", MODEL_FILE)

MODEL_SR = 44100
CHUNK = 352800        # audio.chunk_size from the model yaml (8 s at 44.1 kHz)
NUM_OVERLAP = 2       # inference.num_overlap

# model: section of dereverb_mel_band_roformer_anvuew.yaml (inlined because the
# yaml uses !!python/tuple tags; the mono checkpoint shares this config)
MODEL_CONFIG = dict(
    dim=384, depth=6, stereo=True, num_stems=1,
    time_transformer_depth=1, freq_transformer_depth=1, num_bands=60,
    dim_head=64, heads=8, attn_dropout=0, ff_dropout=0, flash_attn=True,
    dim_freqs_in=1025, sample_rate=MODEL_SR,
    stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048, stft_normalized=False,
    mask_estimator_depth=2, multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
    multi_stft_hop_size=147, multi_stft_normalized=False,
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_checkpoint(path=DEFAULT_CKPT, log=print):
    """Download the checkpoint (about 900 MB) on first use and verify its hash."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    log(f"downloading de-reverb model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    digest = _sha256(tmp)
    if digest != MODEL_SHA256:
        os.remove(tmp)
        raise RuntimeError(f"de-reverb checkpoint hash mismatch: got {digest}")
    os.replace(tmp, path)
    return path


class Dereverb:
    def __init__(self, ckpt=None, device=None, log=print):
        import torch
        from vendor.msst import MelBandRoformer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = ensure_checkpoint(ckpt or os.environ.get("CV_DEREVERB_CKPT", DEFAULT_CKPT), log)
        t0 = time.time()
        model = MelBandRoformer(**MODEL_CONFIG)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()
        log(f"de-reverb model loaded in {time.time() - t0:.1f}s on {self.device}")

    def __call__(self, audio, sr):
        """Return the de-reverberated mono signal at `sr`, same length as `audio`.

        `audio` is (n,) or (n, channels) float; channels are averaged first.
        """
        import librosa
        torch = self.torch

        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        n = len(mono)
        x = mono if sr == MODEL_SR else librosa.resample(mono, orig_sr=sr, target_sr=MODEL_SR)
        y = self._demix(np.stack([x, x]))
        y = y.mean(axis=0)
        if sr != MODEL_SR:
            y = librosa.resample(y, orig_sr=MODEL_SR, target_sr=sr)
        return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)

    def _demix(self, x):
        """MSST style chunked overlap-add with linear fades and reflect padded borders."""
        torch = self.torch
        step = CHUNK // NUM_OVERLAP
        fade = CHUNK // 10
        win = torch.ones(CHUNK)
        win[:fade] = torch.linspace(0, 1, fade)
        win[-fade:] = torch.linspace(1, 0, fade)

        mix = torch.from_numpy(x)
        border = CHUNK - step
        n0 = mix.shape[-1]
        padded = n0 > 2 * border
        if padded:
            mix = torch.nn.functional.pad(mix, (border, border), mode="reflect")
        n = mix.shape[-1]
        out = torch.zeros_like(mix)
        weight = torch.zeros(n)

        with torch.inference_mode():
            i = 0
            while i < n:
                part = mix[:, i:i + CHUNK]
                ln = part.shape[-1]
                if ln < CHUNK:
                    mode = "reflect" if ln > CHUNK // 2 + 1 else "constant"
                    part = torch.nn.functional.pad(part, (0, CHUNK - ln), mode=mode)
                y = self.model(part[None].to(self.device))[0].float().cpu()
                w = win.clone()
                if i == 0:
                    w[:fade] = 1
                if i + step >= n:
                    w[-fade:] = 1
                out[:, i:i + ln] += y[..., :ln] * w[:ln]
                weight[i:i + ln] += w[:ln]
                i += step

        out = out / weight.clamp_min(1e-8)
        if padded:
            out = out[:, border:-border]
        return out.numpy()


def speech_rms(x, sr):
    """RMS over the louder half of 10 ms frames, i.e. the level of the speech itself."""
    hop = max(1, sr // 100)
    frames = x[:len(x) // hop * hop].reshape(-1, hop)
    r = np.sqrt((frames ** 2).mean(axis=1))
    loud = r[r > np.percentile(r, 50)]
    return float(np.sqrt(np.mean(loud ** 2))) if loud.size else 0.0


def match_loudness(processed, reference, sr, peak=0.97):
    """Scale `processed` so its speech level matches `reference`, then cap peaks."""
    ref, cur = speech_rms(reference, sr), speech_rms(processed, sr)
    out = processed * (ref / cur) if cur > 0 else processed
    top = np.abs(out).max() if out.size else 0.0
    if top > peak:
        out = out * (peak / top)
    return out.astype(np.float32)
