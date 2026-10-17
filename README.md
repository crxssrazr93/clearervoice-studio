# ClearerVoice Studio

A LAN web UI (Streamlit) around [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
for cleaning up speech recordings. It runs on the razr93 box (`192.168.2.186`, RTX 3060) at
**http://192.168.2.186:8501**.

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
* **Speed on the RTX 3060:** about 0.35 s of processing per second of audio, plus 4 s to load the model.

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

## Layout on the server

```
~/clearervoice/
  streamlit_app.py  worker.py  cleanup.sh
  .venv/            Python 3.12 venv (see requirements.lock.txt)
  checkpoints/      ClearVoice model weights (downloaded on first use)
  temp/             uploads and *_output/ deliverables
  jobs/             pending/ and done/ queue
~/.config/systemd/user/
  clearervoice.service            the web UI
  clearervoice-cleanup.service    daily prune
  clearervoice-cleanup.timer
```

## Deploy

Copy the code to the server, install the de-reverb extras once, and restart the UI:

```
rsync -a --exclude .git --exclude systemd ./ razr93@192.168.2.186:clearervoice/
ssh razr93@192.168.2.186 'cd ~/clearervoice; ~/.local/bin/uv pip install --python .venv/bin/python --no-deps -r requirements-dereverb.txt'
ssh razr93@192.168.2.186 'systemctl --user restart clearervoice'
```

A running worker keeps its old code until it idles out (30 s), so wait for the queue to drain
before testing a change.

`requirements.lock.txt` is a `uv pip freeze` of the working venv, for rebuilding it from scratch.

## License

Apache License 2.0 (see `LICENSE`). Third-party code and models, including the GPL-3.0 licensed
de-reverb weights that are downloaded at runtime, are listed in `NOTICE`.
