#!/bin/bash
# Double-click to start Audio Studio. It opens in your browser; close this window to stop it.
cd "$(dirname "$0")"
export PATH="$HOME/.local/ffmpeg-bin:$PATH"
lsof -ti tcp:8765 | xargs kill 2>/dev/null
echo "Audio Studio is running at http://127.0.0.1:8765 — keep this window open while you use it."
.venv/bin/python app.py 2>&1 | grep -v "Warning\|fatal: Needed\|from torchaudio\|AudioMetaData"
