"""Audio Studio: local web app (http://127.0.0.1:8765). Runs offline on this Mac.   .venv/bin/python app.py"""
import json, shutil, subprocess, threading, webbrowser
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import engine
from live import Live

ROOT = Path(__file__).parent
LIB = ROOT / "library"
EXPORTS = Path.home() / "Desktop" / "Audio Studio Exports"
MINE = ROOT / "my_presets.json"
app = FastAPI()
lock = threading.Lock()                       # one render at a time keeps memory low

def info(path):
    dur, vid = engine.probe(path)
    if dur <= 0: raise HTTPException(400, "This file has no audio I can read.")
    return {"path": str(path), "name": Path(path).name, "duration": dur, "video": vid, "peaks": engine.peaks(path)}

@app.get("/")
def index(): return HTMLResponse((ROOT / "web" / "index.html").read_text())

@app.post("/api/open")
def open_file():
    r = subprocess.run(["osascript", "-e", 'POSIX path of (choose file with prompt "Choose a video or audio file")'], capture_output=True, text=True)
    if r.returncode: return {"cancelled": True}
    return info(r.stdout.strip())

@app.get("/api/info")
def file_info(path: str):
    if not Path(path).is_file(): raise HTTPException(404, "File not found.")
    return info(path)

@app.post("/api/upload")
def upload(f: UploadFile):
    d = engine.WORK / "uploads"; d.mkdir(exist_ok=True); p = d / Path(f.filename).name
    with open(p, "wb") as o: shutil.copyfileobj(f.file, o)
    if p.suffix.lower() in (".webm", ".ogg", ".opus"):          # browser mic recordings carry no duration: turn them into WAV first
        wav = p.with_suffix(".wav")
        subprocess.run([engine.FF, "-y", "-v", "error", "-i", str(p), "-ac", "1", "-ar", str(engine.SR), str(wav)], check=True)
        p.unlink(); p = wav
    return info(p)

@app.get("/api/setup")
def setup():
    lib = {k: [{"name": p.stem, "path": str(p)} for p in sorted((LIB / k).glob("*")) if p.suffix.lower() in (".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg")]
           for k in ("Music", "Ambience", "My sounds")}
    mine = json.loads(MINE.read_text()) if MINE.exists() else {}
    return {"presets": engine.PRESETS, "mine": mine, "loudness": list(engine.LOUDNESS), "library": lib}

class Job(BaseModel):
    path: str
    params: dict
    start: float = 0
    length: float = 20
    fmt: str = "same"

@app.post("/api/preview")
def preview(j: Job):
    with lock:
        try:
            after = engine.render(j.path, j.params, j.start, j.length)
            before = engine.original_window(j.path, j.start, j.length)
        except subprocess.CalledProcessError as e:
            raise HTTPException(500, (e.stderr or b"").decode()[-400:])
    return {"before": f"/api/file/{before.name}", "after": f"/api/file/{after.name}"}

@app.get("/api/auto")
def auto(path: str):
    with lock: vals, notes = engine.analyze(path)
    return {"values": vals, "notes": notes}

@app.get("/api/file/{name}")
def file(name: str):
    p = engine.WORK / Path(name).name
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p)

@app.get("/api/sound")
def sound(path: str):                        # play a library sound in the browser (library folder only)
    p = Path(path).resolve()
    if LIB.resolve() not in p.parents or not p.exists(): raise HTTPException(404)
    return FileResponse(p)

@app.post("/api/export")
def export(j: Job):
    src = Path(j.path); _, vid = engine.probe(src)
    ext = {"mp4": ".mp4", "wav": ".wav", "mp3": ".mp3", "m4a": ".m4a"}.get(j.fmt) or (".mp4" if vid else src.suffix.lower() if src.suffix.lower() in (".wav", ".mp3", ".m4a") else ".m4a")
    if ext == ".mp4" and not vid: ext = ".m4a"
    EXPORTS.mkdir(exist_ok=True)
    out = EXPORTS / f"{src.stem} - enhanced{ext}"; n = 2
    while out.exists(): out = EXPORTS / f"{src.stem} - enhanced {n}{ext}"; n += 1
    with lock:
        try: engine.render(src, j.params, out=out)
        except subprocess.CalledProcessError as e: raise HTTPException(500, (e.stderr or b"").decode()[-400:])
    return {"out": str(out), "name": out.name}

class Reveal(BaseModel):
    path: str

@app.post("/api/reveal")
def reveal(r: Reveal):
    p = Path(r.path)
    if p.exists(): subprocess.run(["open", "-R", str(p)])
    return {}

@app.post("/api/open-my-sounds")
def open_my_sounds(): subprocess.run(["open", str(LIB / "My sounds")]); return {}

class Preset(BaseModel):
    name: str
    params: dict

@app.post("/api/preset")
def save_preset(p: Preset):
    mine = json.loads(MINE.read_text()) if MINE.exists() else {}
    mine[p.name.strip()[:40] or "My preset"] = p.params; MINE.write_text(json.dumps(mine, indent=1, ensure_ascii=False))
    return mine

@app.delete("/api/preset/{name}")
def delete_preset(name: str):
    mine = json.loads(MINE.read_text()) if MINE.exists() else {}
    mine.pop(name, None); MINE.write_text(json.dumps(mine, indent=1, ensure_ascii=False)); return mine

live = Live()

@app.get("/api/live/devices")
def live_devices():
    devs = live.devices()
    bh = next((d["id"] for d in devs if "BlackHole" in d["name"] and d["out"]), None)
    return {"devices": devs, "blackhole": bh, "running": live.running}

class LiveStart(BaseModel):
    mic: int
    out: int
    monitor: int | None = None
    params: dict = {}

@app.post("/api/live/start")
def live_start(j: LiveStart):
    try: live.start(j.mic, j.out, j.monitor, j.params)
    except Exception as e: live.stop(); raise HTTPException(500, f"Couldn't start: {e}")
    return {"running": True}

@app.post("/api/live/stop")
def live_stop(): live.stop(); return {"running": False}

@app.post("/api/live/params")
def live_params(p: dict): live.set_params(p); return {}

@app.get("/api/live/status")
def live_status(): return {"running": live.running, "levels": live.levels, "error": live.err}

if __name__ == "__main__":
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:8765")).start()
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
