# ClearerVoice Studio

A LAN web UI (Streamlit) around [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
for cleaning up speech recordings. It runs on the razr93 box (`192.168.2.186`, RTX 3060) at
**http://192.168.2.186:8501**.

## Tasks

* **Speech Enhancement.** Removes background noise. Models: `MossFormer2_SE_48K` (default, full band),
  `FRCRN_SE_16K`, `MossFormerGAN_SE_16K`. Optional Silero VAD preprocessing.
* **Speech Separation.** Splits a mix of speakers (`MossFormer2_SS_16K`).
* **Target Speaker Extraction.** Audio visual extraction from a video (`AV_MossFormer2_TSE_16K`).

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

Copy the code to the server and restart the UI:

```
scp streamlit_app.py worker.py cleanup.sh razr93@192.168.2.186:clearervoice/
ssh razr93@192.168.2.186 'systemctl --user restart clearervoice'
```

`requirements.lock.txt` is a `uv pip freeze` of the working venv, for rebuilding it from scratch.

## License

Apache License 2.0 (see `LICENSE`). Third-party code and models, including the GPL-3.0 licensed
de-reverb weights that are downloaded at runtime, are listed in `NOTICE`.
