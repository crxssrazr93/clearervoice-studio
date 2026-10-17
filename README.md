# ClearerVoice Studio

A self-hosted web UI (Streamlit) around [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
for cleaning up speech recordings on a single NVIDIA GPU machine, for example a box on your LAN.
GPU jobs run in a separate worker process that exits when idle, so the card is only held while
something is being processed.

## Tasks

* **Speech Enhancement.** Removes background noise. Models: `MossFormer2_SE_48K` (default, full band),
  `FRCRN_SE_16K`, `MossFormerGAN_SE_16K`. Optional Silero VAD preprocessing, and an optional
  **Remove room echo** step (see below).
* **Speech Separation.** Splits a mix of speakers (`MossFormer2_SS_16K`).
* **Target Speaker Extraction.** Audio visual extraction from a video (`AV_MossFormer2_TSE_16K`).

## Remove room echo (de-reverb)

The ClearVoice enhancement models clean up noise but leave room reverb almost untouched: the echo
you hear right after each word only drops 1 to 3 dB. Ticking **Remove room echo** runs a dedicated
de-reverb model first, then the selected enhancement model, then matches the loudness back to the
original upload.

* **Model:** anvuew's mono MelBand-RoFormer,
  `dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt` from
  [anvuew/dereverb_mel_band_roformer](https://huggingface.co/anvuew/dereverb_mel_band_roformer).
  About 900 MB. It downloads to `checkpoints/dereverb/` on first use and is checked against a pinned
  SHA-256. Set `CV_DEREVERB_CKPT` to use a copy elsewhere.
* **Model code:** vendored in `vendor/msst/` from ZFTurbo's Music-Source-Separation-Training (MIT).
* **Result on three real talking head recordings:** the echo tail 50 to 150 ms after each word went from
  about -8.5 dB to about -19 dB (relative to the speech before it), versus about -10.5 dB with
  enhancement alone. Speech level is unchanged after loudness matching.
* **Speed on an RTX 3060:** about 0.33 s of processing per second of audio, plus 4 s to load the model.

Things to know:

* **Speech only.** Background music is treated as echo and largely removed.
* The model was trained on mono speech and treats any left/right difference as reverb, so input is
  folded to mono internally. Output is mono, like the other enhancement outputs.
* Do not swap in the `sdr_19.1729` or `less_aggressive` checkpoints from the same repo. They were
  trained under a data loading bug (documented in the model README) and pass reverberant speech
  through unchanged.
* The extra packages are in `requirements-dereverb.txt`. Install them with `--no-deps`, otherwise uv
  pulls a second copy of torch.

## Architecture

* `streamlit_app.py` is the UI. It never touches CUDA. It writes job files into `jobs/pending/`
  and polls `jobs/done/` for results.
* `worker.py` owns the GPU. The UI starts it on submit, it drains the queue FIFO with models cached
  in memory, and it exits after `CV_IDLE_TIMEOUT` seconds (default 30) with an empty queue. Process
  exit is what releases the CUDA context, so the card is free when nobody is using the app.
* `cleanup.sh` prunes uploads and job records older than `CV_RETAIN_DAYS` (default 7). Outputs in
  `temp/*_output/` are never deleted.

## Setup

Requirements: Linux, an NVIDIA GPU with a recent driver, Python 3.12 and ffmpeg.

```
git clone https://github.com/crxssrazr93/clearervoice-studio.git ~/clearervoice
cd ~/clearervoice
python3.12 -m venv .venv
.venv/bin/pip install clearvoice streamlit silero-vad
.venv/bin/pip install --no-deps -r requirements-dereverb.txt
.venv/bin/streamlit run streamlit_app.py
```

Then open `http://<host>:8501`. ClearVoice model weights download into `checkpoints/` the first
time each model is used. `requirements.lock.txt` lists the exact package versions of a known
working install (torch 2.13 with CUDA 13).

### Run as a service

The units in `systemd/` are systemd user units and expect the app at `~/clearervoice`:

```
cp systemd/* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now clearervoice.service clearervoice-cleanup.timer
```

To keep the UI running after you log out, enable lingering once with `loginctl enable-linger`.
After updating the code, restart `clearervoice.service`. A worker that is still running keeps the
old code until it idles out (`CV_IDLE_TIMEOUT`, 30 s by default).

### Layout

```
streamlit_app.py  worker.py  cleanup.sh  dereverb.py  vendor/
.venv/            Python venv
checkpoints/      model weights (downloaded on first use)
temp/             uploads, and *_output/ folders with the results
jobs/             pending/ and done/ job queue
```

## License

Apache License 2.0 (see `LICENSE`). Third-party code and models, including the GPL-3.0 licensed
de-reverb weights that are downloaded at runtime, are listed in `NOTICE`.
