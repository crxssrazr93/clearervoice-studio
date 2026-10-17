"""
ClearerVoice Studio -- queue-based UI.

This process never touches CUDA. It writes job files and polls for results;
worker.py owns the GPU and exits 30s after the queue empties, which is what
actually frees the card. The UI stays up the whole time.
"""
import os
import json
import time
import uuid
import fcntl
import subprocess

import streamlit as st

APP = os.path.dirname(os.path.abspath(__file__))
TEMP = os.path.join(APP, "temp")
JOBS = os.path.join(APP, "jobs")
PENDING = os.path.join(JOBS, "pending")
DONE = os.path.join(JOBS, "done")
LOCK = os.path.join(JOBS, "worker.lock")
PY = os.path.join(APP, ".venv", "bin", "python")
WORKER = os.path.join(APP, "worker.py")
IDLE_TIMEOUT = int(os.environ.get("CV_IDLE_TIMEOUT", "30"))

for d in (TEMP, JOBS, PENDING, DONE):
    os.makedirs(d, exist_ok=True)

st.set_page_config(page_title="ClearerVoice Studio", layout="wide")


# --- worker plumbing -------------------------------------------------------
def worker_running():
    """True if a worker holds the lock (i.e. the GPU is currently attached)."""
    try:
        f = open(LOCK, "a+")
    except OSError:
        return False
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        f.close()


def ensure_worker():
    """Start a worker if none is running. A duplicate exits itself via flock."""
    if worker_running():
        return
    subprocess.Popen(
        [PY, WORKER],
        cwd=APP,
        start_new_session=True,
        stdout=open(os.path.join(APP, "worker.log"), "a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "CV_IDLE_TIMEOUT": str(IDLE_TIMEOUT)},
    )
    for _ in range(40):
        if worker_running():
            return
        time.sleep(0.25)


def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    os.makedirs(TEMP, exist_ok=True)
    path = os.path.join(TEMP, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def submit(task, model, input_path, output_dir, **extra):
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "task": task, "model": model,
           "input_path": input_path, "output_dir": output_dir, **extra}
    tmp = os.path.join(PENDING, jid + ".json.tmp")
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, os.path.join(PENDING, jid + ".json"))
    ensure_worker()
    return jid


def wait_for(jids, status_box):
    """Poll until every job id has a result. Returns {jid: result}."""
    results = {}
    started = time.time()
    while len(results) < len(jids):
        for jid in jids:
            if jid in results:
                continue
            done_path = os.path.join(DONE, jid + ".json")
            if os.path.exists(done_path):
                with open(done_path) as f:
                    results[jid] = json.load(f)
        if len(results) == len(jids):
            break

        n_run = sum(1 for j in jids
                    if os.path.exists(os.path.join(DONE, j + ".running")))
        queued = len(jids) - len(results) - n_run
        if not worker_running() and not results:
            status_box.warning("Worker not running yet -- starting it ...")
            ensure_worker()
        status_box.info(
            f"{len(results)}/{len(jids)} done · {n_run} processing · "
            f"{queued} queued · {int(time.time() - started)}s elapsed"
        )
        time.sleep(1.0)
    return results


def render_results(results):
    for jid, res in results.items():
        if res.get("status") == "ok":
            st.success(f"Completed in {res['seconds']}s")
            for out in res.get("outputs", []):
                st.text(out)
                if os.path.exists(out) and os.path.getsize(out) < 300 * 1024 * 1024:
                    st.audio(out)
                    with open(out, "rb") as f:
                        st.download_button("Download", f, file_name=os.path.basename(out),
                                           key=f"dl_{jid}_{os.path.basename(out)}")
                elif os.path.exists(out):
                    st.caption("File too large to preview inline -- path above.")
            if res.get("output_dir"):
                st.text(res["output_dir"])
        else:
            st.error(res.get("error", "unknown error"))
            with st.expander("traceback"):
                st.code(res.get("traceback", ""))


# --- UI --------------------------------------------------------------------
def main():
    st.title("ClearerVoice Studio")

    busy = worker_running()
    st.caption(
        ("🟢 GPU attached (worker running)" if busy
         else "⚪ GPU released (worker idle) — it starts automatically on submit")
        + f" · auto-release after {IDLE_TIMEOUT}s idle"
    )

    tabs = st.tabs(["Speech Enhancement", "Speech Separation", "Target Speaker Extraction"])

    with tabs[0]:
        st.header("Speech Enhancement")
        se_models = ["MossFormer2_SE_48K", "FRCRN_SE_16K", "MossFormerGAN_SE_16K"]
        selected_model = st.selectbox("Select Model", se_models)
        enable_dereverb = st.checkbox(
            "Remove room echo (de-reverb)", key="se_dereverb",
            help="Runs a MelBand-RoFormer de-reverb model before enhancement. "
                 "The enhancement models alone remove noise but barely touch room echo. "
                 "Speech only: background music would be treated as echo. "
                 "First use downloads about 900 MB.")
        enable_vad = st.checkbox("Enable VAD Preprocessing", key="se_vad")
        uploaded = st.file_uploader("Upload Audio File(s)", type=["wav"],
                                    key="se", accept_multiple_files=True)

        if st.button("Start Processing", key="se_process"):
            if uploaded:
                out_dir = os.path.join(TEMP, "speech_enhancement_output")
                jids = [submit("speech_enhancement", selected_model,
                               save_uploaded_file(u), out_dir, enable_vad=enable_vad,
                               dereverb=enable_dereverb)
                        for u in uploaded]
                status_box = st.empty()
                results = wait_for(jids, status_box)
                status_box.empty()
                render_results(results)
            else:
                st.error("Please upload an audio file first")

    with tabs[1]:
        st.header("Speech Separation")
        uploaded = st.file_uploader("Upload Mixed Audio File", type=["wav", "avi"], key="ss")
        if st.button("Start Separation", key="ss_process"):
            if uploaded is not None:
                out_dir = os.path.join(TEMP, "speech_separation_output")
                jid = submit("speech_separation", "MossFormer2_SS_16K",
                             save_uploaded_file(uploaded), out_dir)
                status_box = st.empty()
                results = wait_for([jid], status_box)
                status_box.empty()
                render_results(results)
            else:
                st.error("Please upload an audio file first")

    with tabs[2]:
        st.header("Target Speaker Extraction")
        uploaded = st.file_uploader("Upload Video File", type=["mp4", "avi"], key="tse")
        if st.button("Start Extraction", key="tse_process"):
            if uploaded is not None:
                out_dir = os.path.join(TEMP, "videos_tse_output")
                jid = submit("target_speaker_extraction", "AV_MossFormer2_TSE_16K",
                             save_uploaded_file(uploaded), out_dir)
                status_box = st.empty()
                results = wait_for([jid], status_box)
                status_box.empty()
                render_results(results)
            else:
                st.error("Please upload a video file first")


if __name__ == "__main__":
    main()
