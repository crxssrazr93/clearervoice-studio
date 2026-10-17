#!/usr/bin/env python3
"""
ClearerVoice GPU worker.

Owns the GPU. Drains jobs/pending/ FIFO, keeps models cached in memory so
back-to-back jobs don't pay the load cost, and EXITS after CV_IDLE_TIMEOUT
seconds with an empty queue. Process exit is what actually releases the CUDA
context -- torch cannot free it in-process.

Single instance enforced with flock; a second copy exits silently.
"""
import os
import sys
import json
import time
import glob
import fcntl
import traceback

import numpy as np
import soundfile as sf

APP = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(APP, "jobs")
PENDING = os.path.join(JOBS, "pending")
DONE = os.path.join(JOBS, "done")
LOCK = os.path.join(JOBS, "worker.lock")
IDLE_TIMEOUT = float(os.environ.get("CV_IDLE_TIMEOUT", "30"))

for d in (JOBS, PENDING, DONE):
    os.makedirs(d, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# --- single instance -------------------------------------------------------
_lock = open(LOCK, "a+")
try:
    fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(0)
_lock.seek(0)
_lock.truncate()
_lock.write(str(os.getpid()))
_lock.flush()


# --- model cache -----------------------------------------------------------
_models = {}


def get_model(task, model_name):
    key = (task, model_name)
    if key not in _models:
        from clearvoice import ClearVoice
        t0 = time.time()
        log(f"loading {model_name} for {task} ...")
        _models[key] = ClearVoice(task=task, model_names=[model_name])
        log(f"loaded in {time.time() - t0:.1f}s")
    return _models[key]


# --- task handlers ---------------------------------------------------------
def run_speech_enhancement(job):
    input_path = job["input_path"]
    model_name = job["model"]
    enable_vad = bool(job.get("enable_vad", False))
    out_dir = job["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(input_path))[0]
    suffix = "_vad" if enable_vad else ""
    output_path = os.path.join(out_dir, f"output_{model_name}{suffix}_{stem}.wav")

    speech_ts = None
    orig_audio = orig_sr = None

    if enable_vad:
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
        orig_audio, orig_sr = sf.read(input_path)
        if orig_audio.ndim > 1:
            orig_audio = orig_audio.mean(axis=1)
        vad_model = load_silero_vad()
        vad_wav = read_audio(input_path)
        timestamps = get_speech_timestamps(vad_wav, vad_model, sampling_rate=16000)
        scale = orig_sr / 16000.0
        speech_ts = [{"start": int(t["start"] * scale), "end": int(t["end"] * scale)}
                     for t in timestamps]
        segs = [orig_audio[d["start"]:d["end"]] for d in speech_ts]
        speech_input = np.concatenate(segs) if segs else np.array([])
        seg_path = os.path.join(out_dir, f"_vadsegs_{stem}.wav")
        sf.write(seg_path, speech_input, orig_sr)
        model_input = seg_path
    else:
        model_input = input_path

    cv = get_model("speech_enhancement", model_name)
    output_wav = cv(input_path=model_input, online_write=False)

    if enable_vad:
        model_sr = cv.models[0].args.sampling_rate
        enhanced = np.squeeze(output_wav)
        if model_sr != orig_sr and enhanced.size:
            import librosa
            enhanced = librosa.resample(enhanced, orig_sr=model_sr, target_sr=orig_sr)
        full = np.zeros_like(orig_audio)
        idx = 0
        for d in speech_ts:
            s, e = d["start"], d["end"]
            chunk = enhanced[idx: idx + (e - s)]
            # defensive: model output length can differ from the requested slice
            full[s: s + len(chunk)] = chunk
            idx += (e - s)
        sf.write(output_path, full, orig_sr, format="WAV", subtype="PCM_16")
    else:
        cv.write(output_wav, output_path=output_path)

    return {"outputs": [output_path]}


def run_speech_separation(job):
    input_path = job["input_path"]
    model_name = job.get("model", "MossFormer2_SS_16K")
    out_dir = job["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if input_path.lower().endswith(".avi"):
        import subprocess
        audio_path = os.path.splitext(input_path)[0] + ".wav"
        subprocess.call(["ffmpeg", "-y", "-i", input_path, "-vn",
                         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                         audio_path])
        input_path = audio_path

    stem = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(out_dir, f"output_{model_name}_{stem}.wav")

    cv = get_model("speech_separation", model_name)
    output_wav = cv(input_path=input_path, online_write=False)
    cv.write(output_wav, output_path=output_path)
    return {"outputs": [output_path], "output_dir": out_dir}


def run_tse(job):
    input_path = job["input_path"]
    model_name = job.get("model", "AV_MossFormer2_TSE_16K")
    out_dir = job["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    cv = get_model("target_speaker_extraction", model_name)
    cv(input_path=input_path, online_write=True, output_path=out_dir)
    return {"outputs": [], "output_dir": out_dir}


HANDLERS = {
    "speech_enhancement": run_speech_enhancement,
    "speech_separation": run_speech_separation,
    "target_speaker_extraction": run_tse,
}


# --- main loop -------------------------------------------------------------
def main():
    log(f"worker up (pid {os.getpid()}), idle timeout {IDLE_TIMEOUT:.0f}s")
    last_activity = time.time()

    while True:
        pending = sorted(glob.glob(os.path.join(PENDING, "*.json")),
                         key=os.path.getmtime)

        if not pending:
            idle = time.time() - last_activity
            if idle >= IDLE_TIMEOUT:
                log(f"idle {idle:.0f}s with empty queue -- exiting to release the GPU")
                return
            time.sleep(0.5)
            continue

        job_file = pending[0]
        try:
            with open(job_file) as f:
                job = json.load(f)
        except Exception:
            os.remove(job_file)
            continue
        os.remove(job_file)

        jid = job["id"]
        # mark running so the UI can distinguish queued from in-flight
        with open(os.path.join(DONE, jid + ".running"), "w") as f:
            f.write(str(time.time()))

        t0 = time.time()
        log(f"job {jid}: {job['task']} / {job.get('model')} <- {job['input_path']}")
        try:
            handler = HANDLERS[job["task"]]
            result = handler(job)
            result.update({"status": "ok", "seconds": round(time.time() - t0, 2)})
            log(f"job {jid}: ok in {result['seconds']}s")
        except Exception as e:
            result = {"status": "error", "error": str(e),
                      "traceback": traceback.format_exc(),
                      "seconds": round(time.time() - t0, 2)}
            log(f"job {jid}: FAILED -- {e}")

        tmp = os.path.join(DONE, jid + ".json.tmp")
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, os.path.join(DONE, jid + ".json"))
        try:
            os.remove(os.path.join(DONE, jid + ".running"))
        except FileNotFoundError:
            pass

        last_activity = time.time()


if __name__ == "__main__":
    main()
