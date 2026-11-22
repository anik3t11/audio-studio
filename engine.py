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
VERSION = 5                      # bump when the sound changes, so cached previews are rebuilt

PRESETS = {
    "Cinematic narration": dict(noise=70, clarity=55, warmth=65, echo=30, space=25, punch=60, loudness="Instagram / YouTube"),
    "Meeting – clear voice": dict(noise=85, clarity=60, warmth=50, echo=50, space=0, punch=45, loudness="Meeting / podcast"),
    "Warm podcast": dict(noise=60, clarity=45, warmth=75, echo=40, space=0, punch=70, loudness="Meeting / podcast"),
    "Outdoor / noisy room": dict(noise=95, clarity=65, warmth=45, echo=20, space=0, punch=55, loudness="Instagram / YouTube"),
    "Natural (light touch)": dict(noise=35, clarity=40, warmth=50, echo=10, space=0, punch=30, loudness="Instagram / YouTube"),
}
LOUDNESS = {"Instagram / YouTube": -14, "Meeting / podcast": -16, "Audiobook (ACX)": -18, "Film (quieter)": -20, "TV broadcast (EBU R128)": -23}
TRUE_PEAK = {"Audiobook (ACX)": -3.5}                     # ACX wants peaks under −3 dB; everything else −1.5 dBTP

# Pro voice styles: the standard broadcast chain (low cut → cut mud → compress → de-ess → presence/air → loudness + peak limit),
# tuned to how each kind of voice is usually delivered. Sliders sit at 50 = exactly the style, and move it from there.
STYLES = {
    "Radio presenter":   dict(hp=85, low_f=150, low=4,   chest=1.5, mud_f=300, mud=-3.5, pres_f=4000, pres=4,   air=3,   deess=0.6,  thr=-26, ratio=5,
                              about="Close-mic broadcast sound: full low end, forward voice, tightly controlled level."),
    "Podcast standard":  dict(hp=80, low_f=180, low=1.5, chest=0,   mud_f=300, mud=-3,   pres_f=3500, pres=3,   air=2,   deess=0.5,  thr=-22, ratio=3,
                              about="The common podcast chain, delivered at −16 LUFS (Apple / Spotify spec)."),
    "Audiobook (ACX-ready)": dict(hp=80, low_f=180, low=1, chest=0, mud_f=300, mud=-2,   pres_f=3000, pres=2,   air=1,   deess=0.45, thr=-20, ratio=2.2,
                              about="Natural and even. Meets Audible ACX: −18 to −23 dB RMS, peaks under −3 dB, very quiet background."),
    "YouTube creator":   dict(hp=90, low_f=160, low=2,   chest=0,   mud_f=350, mud=-3.5, pres_f=4500, pres=5,   air=4,   deess=0.65, thr=-24, ratio=4,
                              about="Bright, upfront and punchy, at −14 LUFS for YouTube and Instagram."),
    "Movie trailer (deep)": dict(hp=55, low_f=110, low=6, chest=2,   mud_f=400, mud=-4,   pres_f=3000, pres=2,   air=0,   deess=0.5,  thr=-28, ratio=6,
                              about="Big, deep voice with heavy compression and a touch of room."),
    "Storyteller (warm)": dict(hp=70, low_f=180, low=3,  chest=1,   mud_f=350, mud=-2.5, pres_f=3000, pres=2.5, air=1.5, deess=0.5,  thr=-22, ratio=3,
                              about="Warm and intimate with a soft top and a small room, for narration and stories."),
}
STYLE_PRESETS = {
    "Radio presenter":       dict(noise=80, echo=45, space=0,  loudness="Instagram / YouTube"),
    "Podcast standard":      dict(noise=75, echo=40, space=0,  loudness="Meeting / podcast"),
    "Audiobook (ACX-ready)": dict(noise=85, echo=55, space=0,  loudness="Audiobook (ACX)"),
    "YouTube creator":       dict(noise=75, echo=35, space=0,  loudness="Instagram / YouTube"),
    "Movie trailer (deep)":  dict(noise=80, echo=40, space=12, loudness="Instagram / YouTube"),
    "Storyteller (warm)":    dict(noise=70, echo=30, space=18, loudness="Instagram / YouTube"),
}
for _n, _v in STYLE_PRESETS.items(): _v.update(style=_n, clarity=50, warmth=50, punch=50, about=STYLES[_n]["about"])

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
def _df_model():
    """DeepFilterNet loaded once and shared by editing and meeting mode."""
    global _df
    if _df is None:
        from df.enhance import init_df
        _df = init_df(log_level="ERROR")[:2]
    return _df

def denoise(src, amount):
    """DeepFilterNet with an attenuation limit from the slider: 0 = off, 100 = remove everything that isn't voice (cached)."""
    if amount <= 0: return src
    out = WORK / f"dn_{src.stem}_{amount}.wav"
    if out.exists(): return out
    from df.enhance import enhance, load_audio, save_audio
    model, state = _df_model()
    audio, _ = load_audio(str(src), sr=state.sr())
    lim = 6 + amount * 0.54 if amount < 100 else None          # dB of noise reduction allowed: gentle → full
    save_audio(str(out), enhance(model, state, audio, atten_lim_db=lim), state.sr())
    return out

def plan(s, adv):
    """Slider values (+ optional pro style, + advanced overrides) → the voice EQ and compressor, shared by editing and meeting mode.
    bands: (kind, freq, gain dB, q) with kind low/high shelf or peak."""
    st = STYLES.get(s.get("style") or "")
    if st:
        w, c, pu = (s["warmth"] - 50) / 50, (s["clarity"] - 50) / 50, (s["punch"] - 50) / 50
        bands = [["low", st["low_f"], st["low"] + w * 6, 0.8], ["peak", 120, st["chest"] + w * 2, 1], ["peak", st["mud_f"], st["mud"] - max(0, c) * 2, 1.2],
                 ["peak", st["pres_f"], st["pres"] + c * 6, 0.9], ["high", 10000, st["air"] + c * 4, 0.7]]
        out = dict(hp=st["hp"], deess=st["deess"] if s["clarity"] > 20 else 0, thr=st["thr"] - pu * 6, ratio=max(1.2, st["ratio"] * (1 + pu * 0.6)))
        idx = dict(low=0, mid=2, presence=3, air=4)
    else:                                                      # the original slider mapping. Ranges are wide enough to hear.
        w, c = (s["warmth"] - 50) / 50, s["clarity"] / 100
        bands = [["low", 200, w * 9, 0.8], ["peak", 120, w * 3, 1], ["high", 6000, -w * 3, 0.5], ["peak", 400, -1 - c * 4, 1.2],
                 ["peak", 3200, -5 + c * 15, 0.9], ["high", 9000, -3 + c * 8, 0.5]]
        out = dict(hp=80, deess=0.5 if s["clarity"] > 55 else 0, thr=-16 - s["punch"] / 5, ratio=1.5 + s["punch"] / 100 * 5.5)
        idx = dict(low=0, mid=3, presence=4, air=5)
    for k, i in idx.items():
        if k in adv: bands[i][2] = adv[k]
    out.update(hp=adv.get("highpass", out["hp"]), thr=adv.get("comp_threshold", out["thr"]), ratio=adv.get("comp_ratio", out["ratio"]), bands=bands)
    if "deesser" in adv: out["deess"] = 0.5 if adv["deesser"] else 0
    return out

def voice_chain(s, adv):
    """Slider values → ffmpeg filters for the voice (everything except the reverb). adv = advanced overrides (may be empty)."""
    pl = plan(s, adv)
    f = [f"highpass=f={pl['hp']}:poles=2"]
    if s["echo"] > 0:                                          # downward expander: tails of room echo between words drop away
        f.append(f"agate=threshold={0.002 + s['echo'] * 0.0003:.4f}:ratio={1.5 + s['echo'] * 0.05:.2f}:range={max(0.05, 1 - s['echo'] * 0.0095):.3f}:attack=5:release={250 - s['echo']:.0f}")
    for kind, fr, g, q in pl["bands"]:
        f.append(f"bass=g={g:.1f}:f={fr}:w={q}" if kind == "low" else f"treble=g={g:.1f}:f={fr}:w={q}" if kind == "high" else f"equalizer=f={fr}:t=q:w={q}:g={g:.1f}")
    f.append(f"acompressor=threshold={pl['thr']:.0f}dB:ratio={pl['ratio']:.1f}:attack=6:release=160:makeup=2")
    if pl["deess"]:                                            # after the compressor, as in a broadcast chain. ffmpeg's deesser does
        f.append(f"deesser=i={0.35 + pl['deess'] * 0.15:.3f}:m=0.5:f=0.5")   # nothing below i≈0.35 and dulls all the top above ≈0.5
    return ",".join(f)

def speech_gain(path):
    """dB that brings the speech to −20 dBFS before the chain, so compressor thresholds mean the same on every recording."""
    a, _ = sf.read(str(path), dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    fr = SR // 50; n = len(a) // fr
    if n < 10: return 0.0
    e = np.sqrt((a[: n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-9
    loud = e[e >= np.percentile(e, 70)]
    return float(np.clip(-20 - 20 * np.log10(np.sqrt((loud ** 2).mean())), -24, 24))

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
    s["style"] = p.get("style") if p.get("style") in STYLES else None
    return s

def render(path, p, start=None, length=None, out=None):
    """Build the processed audio. p: slider dict + music/ambience choices + 'advanced'. start/length = preview window.
    Returns the WAV path (preview) or the exported file path."""
    s = settings(p); adv = p.get("advanced") or {}
    src = extract(path); clean = denoise(src, int(s["noise"]))
    target = adv.get("lufs", LOUDNESS.get(s["loudness"], -14)); tp = TRUE_PEAK.get(s["loudness"], -1.5)
    dur, has_video = probe(path)
    win = ["-ss", f"{start:.2f}", "-t", f"{length:.2f}"] if start is not None else []
    inputs = [*win, "-i", str(clean)]
    bedlist = []
    for name, level, duck in [(p.get("music"), p.get("music_level", 40), p.get("duck", 60)), (p.get("ambience"), p.get("ambience_level", 30), 0)]:
        if name and Path(name).exists(): bedlist.append((Path(name), level, duck))
    nsc = sum(1 for b in bedlist if b[2] > 0)
    graph = [f"[0:a]volume={speech_gain(clean):.1f}dB,{voice_chain(s, adv)},aformat=channel_layouts=stereo[dry]"]
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
    graph.append(f"{mix},loudnorm=I={target}:TP={tp}:LRA=11,alimiter=limit={10 ** ((tp + 0.5) / 20):.3f}:level=disabled[out]")
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

def analyze(path):
    """Listen to the file and suggest slider values, with plain-language notes. Leaves space, music and loudness to the user."""
    a, _ = sf.read(str(extract(path)), dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    fr = SR // 50; n = len(a) // fr
    if n < 20: return {}, ["The recording is too short to analyse."]
    frames = a[: n * fr].reshape(n, fr); e = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    speech_db = np.percentile(e, 95); snr = speech_db - np.percentile(e, 3)      # calibrated on clean studio voice (~52), laptop mic (~35), fan noise (~26)
    notes, s = [], {}
    if snr > 48:   s["noise"] = 25; notes.append("Very quiet background")
    elif snr > 38: s["noise"] = 45; notes.append("A little background noise")
    elif snr > 28: s["noise"] = 65; notes.append("Noticeable background noise")
    else:          s["noise"] = 90; notes.append("Heavy background noise")
    speech = frames[e > np.percentile(e, 70)].ravel()                       # tone, measured on the speech only
    F = np.abs(np.fft.rfft(speech[: SR * 30])) ** 2; f = np.fft.rfftfreq(min(len(speech), SR * 30), 1 / SR)
    band = lambda lo, hi: 10 * np.log10(F[(f >= lo) & (f < hi)].sum() + 1e-9)
    boom, bright = band(80, 250) - band(250, 1000), band(2000, 5000) - band(250, 1000)   # a good voice ≈ −3 and −12
    s["warmth"] = int(np.clip(50 - (boom + 3) * 3, 30, 75))
    notes.append("Boomy voice, so a bit lighter" if s["warmth"] < 43 else "Thin voice, so warmer" if s["warmth"] > 57 else "Balanced low end")
    s["clarity"] = int(np.clip(50 - (bright + 12) * 2.5, 30, 80))
    notes.append("Muffled, so clarity added" if s["clarity"] > 58 else "Already bright, so clarity kept gentle" if s["clarity"] < 42 else "Clear enough")
    drops = [e[i] - e[i + 1:i + 8].min() for i in range(2, n - 8) if e[i - 1] > speech_db - 8 and e[i] < speech_db - 14]
    echoey = len(drops) >= 15 and np.median(drops) < 4 and snr > 38                  # sound lingers after words, and it isn't just noise
    s["echo"] = 55 if echoey else 25
    notes.append("Echoey room" if echoey else "Little room echo")
    spread = np.std(e[e > np.percentile(e, 60)])                           # loud vs soft words
    s["punch"] = int(np.clip(35 + spread * 6, 30, 75))
    notes.append("Uneven loudness, so evened out" if s["punch"] > 55 else "Steady loudness")
    return s, notes
