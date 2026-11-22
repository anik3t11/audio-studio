"""Builds the built-in sound library: original music beds and ambiences synthesised in code (so they're ours, free for any use).
Each is 60 s, seamlessly loopable, 48 kHz stereo.   .venv/bin/python make_library.py"""
import subprocess, os
from pathlib import Path
import numpy as np
import soundfile as sf

SR, D = 48000, 60
OUT = Path(__file__).parent / "library"; (OUT / "Music").mkdir(parents=True, exist_ok=True); (OUT / "Ambience").mkdir(exist_ok=True)
(OUT / "My sounds").mkdir(exist_ok=True)
FF = os.path.expanduser("~/.local/ffmpeg-bin/ffmpeg")
rng = np.random.default_rng(7)
t = np.arange(SR * D) / SR

def note(n): return 440 * 2 ** ((n - 69) / 12)            # MIDI → Hz

def pad(chords, bar, bright=0.3, detune=0.15):
    """Soft synth pad: each chord note as 3 slightly detuned saw-ish voices, slow attack/release, crossfaded per bar."""
    y = np.zeros((len(t), 2))
    for i, ch in enumerate(chords * int(np.ceil(D / bar / len(chords)))):
        a, b = i * bar, (i + 1) * bar
        if a >= D: break
        env = np.clip(np.minimum((t - a + 1.5) / 2.5, (b + 1.5 - t) / 2.5), 0, 1) ** 2
        idx = env > 0
        for n in ch:
            for k, dt in enumerate((-detune, 0, detune)):
                f = note(n) * 2 ** (dt / 12)
                ph = 2 * np.pi * f * t[idx] + rng.uniform(0, 6.28)
                w = np.sin(ph) + bright * 0.5 * np.sin(2 * ph) + bright * 0.25 * np.sin(3 * ph)
                y[idx, k % 2] += w * env[idx] * 0.05
                y[idx, (k + 1) % 2] += w * env[idx] * 0.03
    return y

def drone(root, depth=0.5):
    y = np.zeros((len(t), 2))
    for mult, amp in ((0.5, 0.5), (1, 0.4), (1.5, 0.15), (2, 0.1)):
        f = note(root) * mult; lfo = 1 + 0.002 * np.sin(2 * np.pi * 0.07 * t * mult)
        y[:, 0] += amp * np.sin(2 * np.pi * f * t * lfo); y[:, 1] += amp * np.sin(2 * np.pi * f * 1.003 * t * lfo + 1)
    return y * 0.12 * (0.8 + 0.2 * np.sin(2 * np.pi * t / 15))[:, None] * depth * 2

def noise(color="pink", n=None):
    n = n or len(t); w = rng.standard_normal(n)
    F = np.fft.rfft(w); f = np.fft.rfftfreq(n, 1 / SR); f[0] = 1
    F /= f ** (0.5 if color == "pink" else 1.0); y = np.fft.irfft(F, n); return y / np.abs(y).max()

def bandpass(x, lo, hi):
    F = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); F[(f < lo) | (f > hi)] = 0; return np.fft.irfft(F, len(x))

def thump(times, f0=55, dec=0.25, amp=0.5):
    y = np.zeros(len(t))
    for s in times:
        i = int(s * SR); L = int(SR * 0.6); tt = np.arange(min(L, len(t) - i)) / SR
        y[i:i + len(tt)] += amp * np.sin(2 * np.pi * f0 * tt * (1 + 0.6 * np.exp(-tt * 30))) * np.exp(-tt / dec)
    return np.stack([y, y], 1)

def bell(times, f0, amp=0.15):
    y = np.zeros(len(t))
    for s in times:
        i = int(s * SR); L = int(SR * 6); tt = np.arange(min(L, len(t) - i)) / SR
        for r, a, d in ((1, 1, 2.5), (2.76, 0.5, 1.2), (5.4, 0.25, 0.6), (8.9, 0.1, 0.3)):
            y[i:i + len(tt)] += amp * a * np.sin(2 * np.pi * f0 * r * tt) * np.exp(-tt / d)
    return np.stack([y, np.roll(y, 300)], 1)

def save(name, y, reverb=0.3, folder="Music"):
    y = y - y.mean(0); x = len(t) - SR * 3                        # loop seam: crossfade the last 3 s into the start
    fade = np.linspace(0, 1, SR * 3)[:, None]; y[: SR * 3] = y[: SR * 3] * fade + y[x:] * (1 - fade); y = y[:x]
    y = y / (np.abs(y).max() + 1e-9) * 0.7
    tmp = OUT / "_tmp.wav"; sf.write(tmp, y, SR)
    out = OUT / folder / f"{name}.m4a"
    rv = f"aecho=0.8:0.85:60|97|131|173:{0.35 * reverb:.2f}|{0.28 * reverb:.2f}|{0.2 * reverb:.2f}|{0.14 * reverb:.2f}," if reverb else ""
    subprocess.run([FF, "-y", "-v", "error", "-i", str(tmp), "-af", f"{rv}loudnorm=I=-20:TP=-2", "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
    tmp.unlink(); print("made", out.relative_to(OUT))

# --- music beds ---
save("Calm wonder (story intro)", pad([[50, 57, 62, 66], [47, 54, 59, 62], [43, 50, 55, 59], [45, 52, 57, 61]], 7.5, 0.2) + drone(38, 0.3), 0.6)
save("Ancient mystery (minor)", pad([[50, 53, 57, 62], [46, 50, 53, 58], [48, 52, 55, 60], [45, 49, 52, 57]], 7.5, 0.35) + drone(38, 0.6), 0.7)
save("Tension rising", drone(36, 0.9) + thump(np.arange(0.5, D, 0.85), 50, 0.2, 0.35) + pad([[48, 51, 55], [48, 51, 56]], 10, 0.5, 0.25) * 0.6, 0.4)
save("Hopeful ending (major)", pad([[48, 55, 60, 64], [53, 57, 60, 65], [45, 52, 57, 60], [55, 59, 62, 67]], 6, 0.25), 0.6)
save("Sad and tender", pad([[45, 52, 57, 60], [41, 48, 53, 57], [48, 55, 60, 64], [43, 50, 55, 59]], 8, 0.15, 0.1), 0.7)
save("Temple dawn (drone + bells)", drone(50, 0.6) + bell(np.arange(2, D, 9.5), note(74)) + bell(np.arange(6.5, D, 13), note(81), 0.08), 0.8)
# --- ambiences ---
r = bandpass(noise("white"), 400, 9000) * 0.25
drops = np.zeros(len(t)); idx = rng.integers(0, len(t) - 2000, 2500)
for i in idx: drops[i:i + 400] += rng.uniform(0.05, 0.3) * np.exp(-np.arange(400) / 60) * rng.standard_normal(400)
save("Rain (soft)", np.stack([r + drops, np.roll(r, 900) + np.roll(drops, 500)], 1), 0.2, "Ambience")
w = noise("brown"); lfo = 0.5 + 0.5 * np.sin(2 * np.pi * t / 11) * np.sin(2 * np.pi * t / 4.3)
save("Wind", np.stack([w * lfo, np.roll(w, 2000) * lfo[::-1]], 1), 0, "Ambience")
cr = np.zeros(len(t))
for s0 in np.arange(0, D, 0.9):
    for j in range(3):
        i = int((s0 + j * 0.06 + rng.uniform(0, 0.3)) * SR) % (len(t) - 3000); tt = np.arange(2400) / SR
        cr[i:i + 2400] += 0.12 * np.sin(2 * np.pi * 4300 * tt) * np.sin(np.pi * tt / tt[-1]) * (np.sin(2 * np.pi * 60 * tt) > 0)
save("Night crickets", np.stack([cr + bandpass(noise(), 100, 2000) * 0.03, np.roll(cr, 7000) + bandpass(noise(), 100, 2000) * 0.03], 1), 0.3, "Ambience")
save("Room tone (quiet)", np.stack([bandpass(noise(), 60, 3000), bandpass(noise(), 60, 3000)], 1) * 0.3, 0, "Ambience")
fire = bandpass(noise("brown"), 50, 900) * 0.4; cr2 = np.zeros(len(t))
for i in rng.integers(0, len(t) - 600, 900): cr2[i:i + 300] += rng.uniform(0.1, 0.6) * rng.standard_normal(300) * np.exp(-np.arange(300) / 40)
save("Fire crackle", np.stack([fire + cr2, np.roll(fire, 1500) + np.roll(cr2, 800)], 1), 0.1, "Ambience")
