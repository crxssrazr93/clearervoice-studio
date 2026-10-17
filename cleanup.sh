#!/usr/bin/env bash
# Prune ClearerVoice intermediates older than $CV_RETAIN_DAYS (default 7).
#
# DELETES: uploads (top level of temp/), VAD intermediates, job records.
# KEEPS:   everything in the *_output/ subdirectories -- those are deliverables
#          and are never removed automatically.
set -uo pipefail

APP="$HOME/clearervoice"
DAYS="${CV_RETAIN_DAYS:-7}"
[ -d "$APP" ] || exit 0

total=0

# Capture find's output into a variable and echo it from bash, so the paths
# reliably reach the journal. find's own -print is block-buffered under
# systemd and gets lost.
prune() {
  local label="$1"; shift
  local out
  out=$(find "$@" -print -delete 2>/dev/null)
  if [ -n "$out" ]; then
    local n
    n=$(printf '%s\n' "$out" | wc -l)
    total=$((total + n))
    # here-string, not a pipeline: a subshell's buffered stdout does not
    # survive systemd's journal stream, so this must run in the main shell.
    while IFS= read -r f; do
      echo "  deleted [$label] $f"
    done <<< "$out"
  fi
}

echo "prune: retaining $DAYS days, started $(date -Is)"

# 1. uploads: files directly in temp/. The *_output/ subdirectories sit below
#    maxdepth 1 and are never reached.
[ -d "$APP/temp" ] && prune upload "$APP/temp" -maxdepth 1 -type f -mtime +"$DAYS"

# 2. VAD and de-reverb intermediates -- written alongside outputs, but pure scratch.
[ -d "$APP/temp" ] && prune vadseg "$APP/temp" -type f -name '_vadsegs_*' -mtime +"$DAYS"
[ -d "$APP/temp" ] && prune dereverb "$APP/temp" -type f -name '_dereverb_*' -mtime +"$DAYS"

# 3. completed job records and stale running markers
[ -d "$APP/jobs/done" ] && prune job "$APP/jobs/done" -type f \
  \( -name '*.json' -o -name '*.running' \) -mtime +"$DAYS"

# 4. orphaned queue entries (only if a worker was killed mid-queue)
[ -d "$APP/jobs/pending" ] && prune queued "$APP/jobs/pending" -type f \
  -name '*.json' -mtime +"$DAYS"

echo "prune: removed $total file(s); temp now $(du -sh "$APP/temp" 2>/dev/null | cut -f1)"
