#!/bin/bash
# Double-click to open the full studio. The engine runs in the menu-bar app (🎙 at the top of the screen); this starts it if needed.
open -a "$HOME/Applications/Audio Studio.app"
for i in $(seq 1 30); do curl -s -o /dev/null http://127.0.0.1:8765/ && break; sleep 1; done
open http://127.0.0.1:8765
