# FastLabel

**FastLabel** is a quick and simple image labeling tool that uses your trained YOLO model to prefill boxes automatically.  
Label faster, edit easier, and export ready-to-train YOLO files — no clutter, no wasted clicks.

---

## 🚀 Features
- **YOLO Prefill / Scan All** — your model draws boxes for you.  
- **Instant editing** — drag, resize, or delete boxes in seconds.  
- **Smooth workflow** — zoom, pan, undo/redo, and multi-select just work.
---

## INFO

- G — toggle grid.
- 1–9 — set active class (1 = first id in sorted class list).
- Alt while drawing/resizing — constrain to square.
- Shift — multi-select; Shift+drag empty → marquee; Shift+drag selected → move group.
- Ctrl while drawing/moving/resizing — snap to image & box edges (shows dashed hint lines).
- Ctrl+drag on empty canvas — pan.
- Mouse wheel — vertical pan; Shift+wheel — horizontal pan; Ctrl+wheel — zoom at cursor; Fit/−/+ buttons also available.
- Right-click — quick class search near cursor; Enter applies; top button labels current selection with the active class.
- Arrow keys — nudge selection; hold Ctrl for fast nudge.
- Delete — remove selection; “🧹 Clear Boxes” shows a confirmation dialog.
- Duplicates — same-class near-identical boxes get a dashed red halo + “DUP” tag.
- YOLO Prefill — enable, pick model, “Prefill (once)” for current image or “Scan All” for batch (progress + cancel). Detected classes auto-created; label files saved under `yoloLabels/`.
- Autosave — when switching images or jumping from Project, labels write to `yoloLabels/<image>.txt`. Use “💾 Save labels” to force a write anytime.

---
## 🎥 Demo
> LATER 

---

## 🧭 Quick Start

```bash
# Install requirements
pip install pillow ultralytics numpy

# Run the app
python fastlabel.py
