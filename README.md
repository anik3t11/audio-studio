# Audio Studio

An offline voice-cleanup and music tool for macOS. Nothing is uploaded anywhere.

- **Editing mode**: AI noise removal (DeepFilterNet) and an ffmpeg voice chain driven by six simple sliders (noise, clarity, warmth, echo, space, punch). It has pro voice styles (radio, podcast, ACX-ready audiobook, YouTube, trailer, storyteller), music/ambience with ducking, loudness targets, a before/after preview, and export. Video exports keep the picture.
- **Meeting mode**: takes your live mic, cleans it, and sends it into [BlackHole 2ch](https://existential.audio/blackhole/). Choose "BlackHole 2ch" as the microphone in Zoom, Meet, Teams or FaceTime.
- **Menu-bar app** (`menubar.py`): turns the clean mic on and off, picks the voice style and microphone, and can start at login.
- **Built-in library**: the music and ambience are synthesised in code (`make_library.py`), so they are free to use, including on monetised channels. Put your own files in `library/My sounds/`.

## Setup

```bash
brew install ffmpeg blackhole-2ch
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python menubar.py      # menu-bar app + studio at http://127.0.0.1:8765
# or just the studio:
.venv/bin/python app.py
```

Exports are saved to `~/Desktop/Audio Studio Exports`.
