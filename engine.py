"""Audio Studio engine: AI noise removal (DeepFilterNet) + a voice chain (ffmpeg) + music/ambience with ducking.
Every control is a 0–100 slider; settings() turns them into real filter values so the UI stays simple."""
import hashlib, json, os, subprocess, warnings
from pathlib import Path
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
FF = os.path.expanduser("~/.local/ffmpeg-bin/ffmpeg")
ROOT = Path(__file__).parent
WORK = ROOT / "work"; WORK.mkdir(exist_ok=True)
SR = 48000
VERSION = 2                      # bump when the sound changes, so cached previews are rebuilt

PRESETS = {
    "Cinematic narration": dict(noise=70, clarity=55, warmth=65, echo=30, space=25, punch=60, loudness="Instagram / YouTube"),
    "Meeting – clear voice": dict(noise=85, clarity=60, warmth=50, echo=50, space=0, punch=45, loudness="Meeting / podcast"),
    "Warm podcast": dict(noise=60, clarity=45, warmth=75, echo=40, space=0, punch=70, loudness="Meeting / podcast"),
    "Outdoor / noisy room": dict(noise=95, clarity=65, warmth=45, echo=20, space=0, punch=55, loudness="Instagram / YouTube"),
    "Natural (light touch)": dict(noise=35, clarity=40, warmth=50, echo=10, space=0, punch=30, loudness="Instagram / YouTube"),
}
LOUDNESS = {"Instagram / YouTube": -14, "Meeting / podcast": -16, "Film (quieter)": -20}

def run(args): subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def probe(path):
    """Duration and whether the file has a video stream (ffmpeg only; ffprobe isn't installed)."""
    err = subprocess.run([FF, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    dur = 0.0
    for line in err.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":"); dur = int(h) * 3600 + int(m) * 60 + float(s)
    has_video = any("Video:" in l and "attached pic" not in l for l in err.splitlines())
    return dur, has_video

def key(*parts): return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]

def extract(path):
    """The file's audio as 48 kHz mono WAV (cached)."""
    st = os.stat(path); out = WORK / f"src_{key(path, st.st_mtime, st.st_size)}.wav"
    if not out.exists(): run([FF, "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), str(out)])
    return out

_df = None
def denoise(src, amount):
    """DeepFilterNet with an attenuation limit from the slider: 0 = off, 100 = remove everything that isn't voice (cached)."""
    global _df
    if amount <= 0: return src
    out = WORK / f"dn_{src.stem}_{amount}.wav"
    if out.exists(): return out
    from df.enhance import enhance, init_df, load_audio, save_audio
    if _df is None: _df = init_df(log_level="ERROR")[:2]
    model, state = _df
    audio, _ = load_audio(str(src), sr=state.sr())
    lim = 6 + amount * 0.54 if amount < 100 else None          # dB of noise reduction allowed: gentle → full
    save_audio(str(out), enhance(model, state, audio, atten_lim_db=lim), state.sr())
    return out

def voice_chain(s, adv):
    """Slider values → ffmpeg filters for the voice (everything except the reverb). adv = advanced overrides (may be empty).
    Ranges are deliberately wide enough to hear: 50 is neutral on warmth, the ends are clearly different."""
    f = [f"highpass=f={adv.get('highpass', 80)}:poles=2"]
    if s["echo"] > 0:                                          # downward expander: tails of room echo between words drop away
        f.append(f"agate=threshold={0.002 + s['echo'] * 0.0003:.4f}:ratio={1.5 + s['echo'] * 0.05:.2f}:range={max(0.05, 1 - s['echo'] * 0.0095):.3f}:attack=5:release={250 - s['echo']:.0f}")
    w = (s["warmth"] - 50) / 50                                # −1 lighter … +1 deeper
    f.append(f"bass=g={adv.get('low', w * 9):.1f}:f=200:w=0.8")                         # ±9 dB low shelf
    f.append(f"equalizer=f=120:t=q:w=1:g={w * 3:.1f}")                                 # chest
    f.append(f"treble=g={-w * 3:.1f}:f=6000")                                          # tilt the top the other way
    c = s["clarity"] / 100                                      # 0 muffled … 1 crisp
    f.append(f"equalizer=f=400:t=q:w=1.2:g={adv.get('mid', -1 - c * 4):.1f}")          # tame boxiness
    f.append(f"equalizer=f=3200:t=q:w=0.9:g={adv.get('presence', -5 + c * 15):.1f}")   # presence −5 … +10 dB
    f.append(f"treble=g={adv.get('air', -3 + c * 8):.1f}:f=9000")                      # air −3 … +5 dB
    if adv.get("deesser", s["clarity"] > 55): f.append("deesser=i=0.5:m=0.5:f=0.5")
    ratio = 1.5 + s["punch"] / 100 * 5.5                        # compressor: gentle 1.5:1 → firm 7:1
    f.append(f"acompressor=threshold={adv.get('comp_threshold', -16 - s['punch'] / 5):.0f}dB:ratio={adv.get('comp_ratio', ratio):.1f}:attack=6:release=160:makeup=2")
    return ",".join(f)

def impulse(space):
    """A stereo room impulse response for afir: decaying filtered noise, 0.5 s (small room) … 2.2 s (hall). Cached per value."""
    out = WORK / f"ir_{space}.wav"
    if out.exists(): return out
    d = 0.5 + space / 100 * 1.7; n = int(SR * d); t = np.arange(n) / SR
    rng = np.random.default_rng(1)
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / d)[:, None]          # −60 dB at the end
    F = np.fft.rfft(ir, axis=0); fr = np.fft.rfftfreq(n, 1 / SR)
    F *= (1 / (1 + (fr / 5000) ** 2))[:, None]                                # darker tail, like a real room
    ir = np.fft.irfft(F, n, axis=0); ir[: int(SR * 0.012)] = 0                # 12 ms pre-delay
    sf.write(out, (ir / np.abs(ir).max() * 0.5).astype(np.float32), SR); return out
def settings(p):
    s = dict(PRESETS["Cinematic narration"]); s.update({k: v for k, v in p.items() if k in s})
    return s

def render(path, p, start=None, length=None, out=None):
    """Build the processed audio. p: slider dict + music/ambience choices + 'advanced'. start/length = preview window.
    Returns the WAV path (preview) or the exported file path."""
    s = settings(p); adv = p.get("advanced") or {}
    src = extract(path); clean = denoise(src, int(s["noise"]))
    target = adv.get("lufs", LOUDNESS.get(s["loudness"], -14))
    dur, has_video = probe(path)
    win = ["-ss", f"{start:.2f}", "-t", f"{length:.2f}"] if start is not None else []
    inputs = [*win, "-i", str(clean)]
    bedlist = []
    for name, level, duck in [(p.get("music"), p.get("music_level", 40), p.get("duck", 60)), (p.get("ambience"), p.get("ambience_level", 30), 0)]:
        if name and Path(name).exists(): bedlist.append((Path(name), level, duck))
    nsc = sum(1 for b in bedlist if b[2] > 0)
    graph = [f"[0:a]{voice_chain(s, adv)},aformat=channel_layouts=stereo[dry]"]
    if s["space"] > 0:                                          # real convolution reverb, mixed under the dry voice
        inputs += ["-i", str(impulse(int(s["space"])))]; ir_idx = sum(1 for a in inputs if a == "-i") - 1
        wet = 0.08 + s["space"] / 100 * 0.42
        graph.append(f"[dry]asplit=2[d1][d2];[d2][{ir_idx}:a]afir=dry=10:wet=10,volume={wet:.2f}[wet];[d1][wet]amix=inputs=2:duration=first:normalize=0[vv]")
    else: graph.append("[dry]anull[vv]")
    graph.append("[vv]" + (f"asplit={1 + nsc}[v]" + "".join(f"[sc{k}]" for k in range(nsc)) if nsc else "anull[v]"))
    beds, k = [], 0
    total = length if length else dur
    for n, (bed, level, duck) in enumerate(bedlist):
        inputs += ["-stream_loop", "-1", *(["-ss", f"{start % max(1, probe(bed)[0]):.2f}"] if start else []), "-i", str(bed)]
        idx = sum(1 for a in inputs if a == "-i") - 1
        db = -42 + level / 100 * 32                             # level slider → −42 … −10 dB
        chain = f"[{idx}:a]aformat=channel_layouts=stereo,aresample={SR},volume={db:.1f}dB,atrim=0:{total:.2f},afade=t=in:d=1.5,afade=t=out:st={max(0, total - 2.5):.2f}:d=2.5"
        if duck > 0:                                            # music dips under the voice and comes back in pauses
            graph.append(f"{chain}[b{n}raw]")
            graph.append(f"[b{n}raw][sc{k}]sidechaincompress=threshold=0.015:ratio={2 + duck / 100 * 10:.1f}:attack=30:release=450:makeup=1[b{n}]"); k += 1
        else: graph.append(f"{chain}[b{n}]")
        beds.append(f"[b{n}]")
    mix = f"[v]{''.join(beds)}amix=inputs={1 + len(beds)}:duration=first:normalize=0" if beds else "[v]anull"
    graph.append(f"{mix},loudnorm=I={target}:TP=-1.5:LRA=11,alimiter=limit=0.89[out]")
    if out is None:
        out = WORK / f"pv_{key(VERSION, path, json.dumps(p, sort_keys=True), start, length)}.wav"
        if not out.exists(): run([FF, "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph), "-map", "[out]", "-ar", str(SR), str(out)])
        return out
    out = Path(out)
    if has_video and out.suffix.lower() in (".mp4", ".mov"):                                               # keep the picture untouched, replace the sound
        run([FF, "-y", "-v", "error", *inputs, "-i", str(path), "-filter_complex", ";".join(graph),
             "-map", f"{sum(1 for a in inputs if a == '-i')}:v:0", "-map", "[out]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-shortest", str(out)])
    else:
        codec = ["-c:a", "pcm_s24le"] if out.suffix == ".wav" else ["-c:a", "aac", "-b:a", "192k"] if out.suffix == ".m4a" else ["-c:a", "libmp3lame", "-q:a", "2"]
        run([FF, "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph), "-map", "[out]", "-ar", str(SR), *codec, str(out)])
    return out

def original_window(path, start, length):
    """The untouched audio for the same window, loudness-matched so before/after compare fairly."""
    out = WORK / f"or_{key(path, start, length)}.wav"
    if not out.exists():
        run([FF, "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", str(extract(path)),
             "-af", "aformat=channel_layouts=stereo,loudnorm=I=-16:TP=-1.5", "-ar", str(SR), str(out)])
    return out

def peaks(path, n=1200):
    """Waveform outline for the UI."""
    a, _ = sf.read(str(extract(path)), dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    step = max(1, len(a) // n); a = a[: step * n].reshape(-1, step) if len(a) >= n else a.reshape(1, -1)
    m = np.abs(a).max(1); return (m / (m.max() + 1e-9)).round(3).tolist()
