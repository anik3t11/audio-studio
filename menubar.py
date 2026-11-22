"""Audio Studio in the menu bar: 🎙 clean mic on/off, voice style and microphone, for FaceTime, Zoom, Meet, WhatsApp…
It also serves the full studio (editing mode) at http://127.0.0.1:8765, sharing the same live engine.
Started by "Audio Studio.app" (in ~/Applications), which can start at login."""
import json, os, threading, time, webbrowser
from pathlib import Path

os.environ["PATH"] = os.path.expanduser("~/.local/ffmpeg-bin") + ":" + os.environ.get("PATH", "")
import rumps, sounddevice as sd, uvicorn
import app as studio, engine

ROOT = Path(__file__).parent
CFG = ROOT / "menubar_settings.json"
URL = "http://127.0.0.1:8765"
AGENT = Path.home() / "Library/LaunchAgents/local.audiostudio.menubar.plist"
BUNDLE = Path.home() / "Applications/Audio Studio.app"
SKIP = ("BlackHole", "Teams Audio")          # virtual devices: never use them as the real mic (that would loop)
live = studio.live
HELP = """1. Keep the 🎙 icon green (clean mic ON).
2. In your call app, choose "BlackHole 2ch" as the microphone:
   • FaceTime: menu bar → Video → BlackHole 2ch
   • Zoom: Settings → Audio → Microphone
   • Google Meet: ⋮ → Settings → Audio
   Or set it once for every app: System Settings → Sound → Input → BlackHole 2ch.
3. Control Center → Mic Mode → Standard (so macOS doesn't process your voice twice).

If people can't hear you, the clean mic is probably OFF: turn it on here, or switch the app back to your normal microphone."""


def load():
    try: return json.loads(CFG.read_text())
    except Exception: return {}


def all_styles():
    """Pro styles first, then quick presets, then the user's own (⭐) presets."""
    mine = json.loads(studio.MINE.read_text()) if studio.MINE.exists() else {}
    return [engine.STYLE_PRESETS, engine.PRESETS, {"⭐ " + k: v for k, v in mine.items()}]


def find_style(name):
    for g in all_styles():
        if name in g: return dict(g[name])
    return dict(engine.STYLE_PRESETS["Podcast standard"])


def refresh_devices():
    """PortAudio only sees devices that existed when it started, so re-scan (only while no stream is open)."""
    if live.running: return
    try: sd._terminate(); sd._initialize()
    except Exception: pass


def mics():
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0 and not any(s in d["name"] for s in SKIP)]


def blackhole():
    return next((i for i, d in enumerate(sd.query_devices()) if "BlackHole" in d["name"] and d["max_output_channels"] > 0), None)


class MenuApp(rumps.App):
    def __init__(self):
        super().__init__("Audio Studio", title="🎙", quit_button=None)
        self.cfg = {"style": "Podcast standard", "mic": None, "on": True, **load()}
        self.busy = False; self.msg = ""; self.using = ""; self.first = True
        self.toggle = rumps.MenuItem("Turn clean mic ON", callback=self.flip)
        self.status = rumps.MenuItem("Starting…")
        self.style_menu = rumps.MenuItem("Voice style")
        self.mic_menu = rumps.MenuItem("Microphone")
        self.login = rumps.MenuItem("Start at login", callback=self.flip_login)
        self.menu = [self.toggle, self.status, None, self.style_menu, self.mic_menu, None,
                     rumps.MenuItem("Open full studio (editing)…", callback=lambda _: webbrowser.open(URL)),
                     rumps.MenuItem("How to use in FaceTime / Zoom…", callback=lambda _: rumps.alert("Using the clean mic", HELP)),
                     self.login, None, rumps.MenuItem("Quit Audio Studio", callback=self.quit)]
        self.login.state = AGENT.exists()
        self.build_styles(); self.build_mics()
        rumps.Timer(self.tick, 1.0).start()

    # ---- settings ----
    def save(self): CFG.write_text(json.dumps(self.cfg, indent=1, ensure_ascii=False))

    def build_styles(self):
        if self.style_menu._menu is not None: self.style_menu.clear()
        for n, g in enumerate(all_styles()):
            if n and g: self.style_menu.add(rumps.separator)
            for name in g:
                it = rumps.MenuItem(name, callback=self.pick_style); it.state = name == self.cfg["style"]; self.style_menu.add(it)

    def build_mics(self):
        if self.mic_menu._menu is not None: self.mic_menu.clear()
        auto = rumps.MenuItem("Automatic (built-in mic)", callback=self.pick_mic); auto.state = not self.cfg["mic"]; self.mic_menu.add(auto)
        for _, name in mics():
            it = rumps.MenuItem(name, callback=self.pick_mic); it.state = name == self.cfg["mic"]; self.mic_menu.add(it)
        self.mic_menu.add(rumps.separator)
        self.mic_menu.add(rumps.MenuItem("Refresh list (after plugging in a mic)", callback=self.refresh))

    def pick_style(self, it):
        self.cfg["style"] = it.title; self.save(); self.build_styles()
        if live.running: live.set_params(find_style(it.title))

    def pick_mic(self, it):
        self.cfg["mic"] = None if it.title.startswith("Automatic") else it.title; self.save(); self.build_mics()
        if live.running: self.start()

    def refresh(self, _):
        was = live.running; live.stop(); refresh_devices(); self.build_mics()
        if was: self.start()

    # ---- clean mic ----
    def choose_mic(self):
        ms = mics()
        for i, n in ms:
            if n == self.cfg["mic"]: return i, n
        for i, n in ms:
            if "MacBook" in n or "Built-in" in n: return i, n
        return ms[0] if ms else (None, None)

    def start(self):
        if self.busy: return
        self.busy = True; self.msg = ""
        def go():
            try:
                live.stop(); refresh_devices()
                mic, name = self.choose_mic(); out = blackhole()
                if out is None: self.msg = "BlackHole 2ch isn't installed"; return
                if mic is None: self.msg = "No microphone found"; return
                live.start(mic, out, None, find_style(self.cfg["style"])); self.using = name
            except Exception as e:
                live.stop(); self.msg = f"Couldn't start: {e}"[:80]
            finally: self.busy = False
        threading.Thread(target=go, daemon=True).start()

    def flip(self, _):
        if live.running: live.stop(); self.cfg["on"] = False
        else: self.cfg["on"] = True; self.start()
        self.save()

    def flip_login(self, it):
        if AGENT.exists(): AGENT.unlink(); it.state = False; return
        AGENT.parent.mkdir(parents=True, exist_ok=True)
        AGENT.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
 <key>Label</key><string>local.audiostudio.menubar</string>
 <key>ProgramArguments</key><array><string>/usr/bin/open</string><string>-a</string><string>{BUNDLE}</string></array>
 <key>RunAtLoad</key><true/>
</dict></plist>
""")
        it.state = True

    def tick(self, _):
        if self.first:                                   # no Dock icon: this is a menu-bar-only app
            self.first = False
            try:
                from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
                NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
            except Exception: pass
            if self.cfg["on"]: self.start()
        # watchdog: the mic went quiet (AirPods disconnected, USB mic unplugged) → restart on whatever mic is there now
        if live.running and not self.busy and time.time() - getattr(live, "last_in", time.time()) > 2.5:
            self.msg = "Mic changed, reconnecting…"; self.start()
        if live.err and not self.busy and self.cfg["on"]:
            self.msg = "Restarting after an error…"; live.err = None; self.start()
        on = live.running
        self.title = "🟢🎙" if on else "🎙"
        self.toggle.title = "Turn clean mic OFF" if on else "Turn clean mic ON"
        style = self.cfg["style"]
        self.status.title = self.msg if self.msg and not on else (f"ON · {style} · {self.using}" if on else ("Starting…" if self.busy else "OFF · callers hear silence on BlackHole"))
        if on and self.msg and not self.busy: self.msg = ""

    def quit(self, _):
        live.stop(); rumps.quit_application()


def serve():
    try: uvicorn.Server(uvicorn.Config(studio.app, host="127.0.0.1", port=8765, log_level="warning")).run()
    except Exception as e: print("studio server:", e)


if __name__ == "__main__":
    threading.Thread(target=serve, daemon=True).start()
    MenuApp().run()
