"""Meeting mode: mic → AI noise removal (DeepFilterNet, in 50 ms chunks with 0.5 s of context) → voice EQ/compressor → a virtual mic (BlackHole).
Zoom/Meet/Teams then pick "BlackHole 2ch" as their microphone. Latency ≈ 100 ms.
Measured: the live chunks remove as much noise as the offline engine (background 47 dB under the voice vs 45 dB)."""
import collections, threading, time
import numpy as np
import sounddevice as sd
from scipy.signal import sosfilt

SR, BLK, CTX, LAG = 48000, 2400, 24000, 960       # 50 ms blocks, 0.5 s context (as clean as 1 s, half the CPU), 20 ms look-ahead trim

def biquad(kind, f, g=0.0, q=0.707):
    """RBJ cookbook biquad → one SOS row."""
    A = 10 ** (g / 40); w = 2 * np.pi * f / SR; c, s = np.cos(w), np.sin(w); al = s / (2 * q)
    if kind == "hp":   b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]; a = [1 + al, -2 * c, 1 - al]
    elif kind == "peak": b = [1 + al * A, -2 * c, 1 - al * A]; a = [1 + al / A, -2 * c, 1 - al / A]
    else:
        sq = 2 * np.sqrt(A) * al
        if kind == "low":
            b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)]
            a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq]
        else:
            b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)]
            a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq]
    return np.array(b + a) / a[0]

class Live:
    def __init__(self):
        self.running = False; self.err = None; self.levels = (0.0, 0.0); self.params = {}
        self.lock = threading.Lock()

    def devices(self):
        out = []
        for i, d in enumerate(sd.query_devices()):
            out.append({"id": i, "name": d["name"], "in": d["max_input_channels"] > 0, "out": d["max_output_channels"] > 0})
        return out

    def set_params(self, p):
        from engine import plan, STYLES
        s = {"noise": 85, "clarity": 60, "warmth": 50, "echo": 40, "punch": 45, **{k: v for k, v in p.items() if isinstance(v, (int, float))}}
        s["style"] = p.get("style") if p.get("style") in STYLES else None
        pl = plan(s, p.get("advanced") or {})                      # the same EQ + compressor as the editing engine
        sos = np.array([biquad("hp", pl["hp"])] + [biquad(k, f, g, q) for k, f, g, q in pl["bands"]])
        with self.lock:
            if getattr(self, "sos", None) is None or self.sos.shape != sos.shape: self.zi = np.zeros((sos.shape[0], 2))
            self.sos, self.s, self.comp = sos, s, (pl["thr"], pl["ratio"])

    def start(self, mic, out, monitor=None, params=None):
        self.stop()
        from engine import _df_model
        self.model, self.state = _df_model()
        self.set_params(params or {})
        self.buf = np.zeros(CTX + BLK + LAG, np.float32); self.env = 0.0; self.gain = 1.0; self.gate = 1.0
        self.q_in = collections.deque(maxlen=20); self.q_out = collections.deque(maxlen=8); self.q_mon = collections.deque(maxlen=8)
        self.err = None; self.running = True
        for q in (self.q_out, self.q_mon): q.append(np.zeros(BLK, np.float32))      # one block of cushion against jitter
        self.streams = [sd.InputStream(device=mic, samplerate=SR, channels=1, blocksize=BLK, dtype="float32", callback=self._in)]
        self.streams.append(sd.OutputStream(device=out, samplerate=SR, channels=2, blocksize=BLK, dtype="float32", callback=self._out(self.q_out)))
        if monitor is not None: self.streams.append(sd.OutputStream(device=monitor, samplerate=SR, channels=2, blocksize=BLK, dtype="float32", callback=self._out(self.q_mon)))
        self.worker = threading.Thread(target=self._work, daemon=True); self.worker.start()
        for st in self.streams: st.start()

    def stop(self):
        self.running = False
        for st in getattr(self, "streams", []):
            try: st.stop(); st.close()
            except Exception: pass
        self.streams = []; self.levels = (0.0, 0.0)

    def _in(self, data, frames, t, status): self.q_in.append(data[:, 0].copy())

    def _out(self, q):
        def cb(data, frames, t, status):
            x = q.popleft() if q else np.zeros(frames, np.float32)
            data[:] = np.repeat(x[:frames, None], 2, 1)
        return cb

    def _work(self):
        import torch
        from df.enhance import enhance
        while self.running:
            if not self.q_in: time.sleep(0.003); continue
            x = self.q_in.popleft()
            try:
                with self.lock: s, sos = self.s, self.sos
                self.buf = np.concatenate([self.buf[BLK:], x])
                if s["noise"] > 0:
                    lim = 6 + s["noise"] * 0.54 if s["noise"] < 100 else None
                    y = enhance(self.model, self.state, torch.from_numpy(self.buf[None]), atten_lim_db=lim).numpy()[0][-BLK - LAG:-LAG]
                else: y = self.buf[-BLK - LAG:-LAG].copy()
                with self.lock: y, self.zi = sosfilt(sos, y, zi=self.zi)
                y = self._dynamics(y.astype(np.float32), s)
                self.levels = (float(np.abs(x).max()), float(np.abs(y).max()))
                for q in (self.q_out, self.q_mon):
                    if len(q) > 4: q.popleft()                           # clock drift over a long call: skip 50 ms rather than lag
                    q.append(y)
            except Exception as e:
                self.err = str(e); self.running = False

    def _dynamics(self, y, s):
        """Per 1 ms: echo gate, compressor and a slow auto-level, then a soft limiter."""
        fr = y.reshape(-1, 48); rms = np.sqrt((fr ** 2).mean(1)) + 1e-7; g = np.empty(len(rms), np.float32)
        thr_gate = 10 ** ((-58 + s["echo"] * 0.25) / 20); thr = 10 ** (self.comp[0] / 20); ratio = self.comp[1]
        for i, r in enumerate(rms):
            self.env = max(r, self.env * 0.97)                       # fast attack, ~30 ms release
            tg = 1.0 if self.env > thr_gate else max(0.1, 1 - s["echo"] / 110)
            self.gate += (tg - self.gate) * (0.3 if tg > self.gate else 0.02)
            over = self.env / thr; c = over ** (1 / ratio - 1) if over > 1 else 1.0
            g[i] = self.gate * c
        speech = rms[rms > 0.003]
        if len(speech):                                              # bring speech toward −18 dBFS RMS, slowly
            want = 0.126 / (np.sqrt((speech ** 2).mean()) * g.mean() + 1e-6)
            self.gain += (np.clip(want, 0.3, 12) - self.gain) * 0.05
        y = (fr * (g * self.gain)[:, None]).ravel()
        return np.tanh(y * 1.1) / 1.1                               # soft limiter, never clips
