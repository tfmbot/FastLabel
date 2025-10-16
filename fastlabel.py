import os, math, time, traceback
from typing import List, Tuple, Optional, Dict, Set

# ---------- switches / layout ----------
SHOW_STATUS_BAR = False     # Set True to show a bottom status bar again
HEADER_STATUS_CHARS = 28    # width (characters) for top-right header status
MAX_HISTORY = 150           # Undo/redo history depth

def log_exc(where: str, ex: BaseException):
    print(f"\n[ERROR] in {where}: {type(ex).__name__} - {ex}")
    traceback.print_exc()
# ---------- deps ----------
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog, colorchooser

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    _YOLO_OK = False

# Dir names
OUTPUT_LBL_DIR = "yoloLabels"

# YOLO model ids — per your request
YOLO_LOCKED_ID   = 0
YOLO_UNLOCKED_ID = 1

DEFAULT_YOLO_MODEL  = r""  # use Browse… to select best.pt
YOLO_CONF_THRESHOLD = 0.25

LEFT_PANEL_WIDTH   = 380
IMAGE_AREA_WIDTH   = 920
IMAGE_AREA_HEIGHT  = 920
STATUS_MAX_CHARS   = 110

HANDLE_SIZE = 8
MIN_SIDE    = 4

# Pan speed (pixels per wheel notch; smaller = slower)
PAN_PIXELS_PER_NOTCH = 30

PALETTE = {
    "bg":        "#1a1f27",
    "panel":     "#222933",
    "card":      "#28303b",
    "canvasbg":  "#141a21",
    "fg":        "#ffffff",
    "muted":     "#c0ccd9",
    "accent":    "#00e6d2",
    "accent2":   "#3dfc8f",
    "warning":   "#ffd166",
    "danger":    "#ff4c4c",
    "outline":   "#3a4a5a",
    "outline2":  "#46586a",
    "hover":     "#304156",
    "crosshair": "#FF4500"
}

EXTRA_COLORS = [
    "#1E90FF", "#8000FF", "#FF1493", "#FFA500",
    "#00FF7F", "#00CED1", "#9400D3", "#FF4500",
    "#FFD700", "#00FA9A", "#4169E1",
]

def ensure_dirs():
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

# ---------- scrollable sidebar ----------
class ScrollableSidebar(ttk.Frame):
    def __init__(self, parent, width):
        super().__init__(parent, style="Sidebar.TFrame")
        SCROLLBAR_W = 18
        canvas_width = max(50, width - SCROLLBAR_W)

        self.canvas = tk.Canvas(self, bg=PALETTE["panel"], width=canvas_width,
                                highlightthickness=0, relief="flat", borderwidth=0)
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.inner = ttk.Frame(self.canvas, style="Sidebar.TFrame")
        self.inner_id = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")

        self.canvas.pack(side=tk.LEFT, fill=tk.Y)
        self.vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._install_global_wheel_router()

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _install_global_wheel_router(self):
        # Capture wheel events app-wide, but route only if pointer is over the sidebar
        self.bind_all("<MouseWheel>", self._wheel_router, add="+")      # Windows/macOS
        self.bind_all("<Button-4>", lambda e: self._wheel_router_linux(e, +1), add="+")  # Linux up
        self.bind_all("<Button-5>", lambda e: self._wheel_router_linux(e, -1), add="+")  # Linux down
    
    def _pointer_in_canvas(self) -> bool:
        try:
            px, py = self.winfo_pointerx(), self.winfo_pointery()
            cx, cy = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            return (cx <= px < cx + cw) and (cy <= py < cy + ch)
        except Exception:
            return False
    
    def _wheel_router(self, e):
        if self._pointer_in_canvas():
            # Reuse existing wheel logic
            self._on_wheel(e)
            return "break"  # stop it from reaching other widgets
    
    def _wheel_router_linux(self, e, direction: int):
        if self._pointer_in_canvas():
            if direction > 0:
                self._on_wheel_linux_up(e)
            else:
                self._on_wheel_linux_down(e)
            return "break"
    

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, _event=None):
        self.canvas.itemconfig(self.inner_id, width=self.canvas.winfo_width())

    # Wheel on Win/Mac
    def _on_wheel(self, e):
        step = -1 if e.delta > 0 else 1
        self.canvas.yview_scroll(step, "units")
    # Wheel on Linux/X11
    def _on_wheel_linux_up(self, _e):   self.canvas.yview_scroll(-1, "units")
    def _on_wheel_linux_down(self, _e): self.canvas.yview_scroll(+1, "units")

    def _bind_mousewheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel_linux_up)
        self.canvas.bind_all("<Button-5>", self._on_wheel_linux_down)

    def _unbind_mousewheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

# ---------- data ----------
class Box:
    __slots__ = ("x1","y1","x2","y2","cls","selected")
    def __init__(self, x1:int, y1:int, x2:int, y2:int, cls:int, selected=False):
        xa, xb = sorted((int(x1), int(x2)))
        ya, yb = sorted((int(y1), int(y2)))
        self.x1, self.y1, self.x2, self.y2 = xa, ya, xb, yb
        self.cls = int(cls)
        self.selected = bool(selected)
    def contains(self, x:int, y:int)->bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    def size_ok(self, min_side:int=MIN_SIDE)->bool:
        return (self.x2-self.x1)>=min_side and (self.y2-self.y1)>=min_side
    def move_by(self, dx:int, dy:int, bounds:Tuple[int,int]):
        iw, ih = bounds
        w = self.x2 - self.x1; h = self.y2 - self.y1
        nx1 = min(max(self.x1 + dx, 0), max(iw - 1 - w, 0))
        ny1 = min(max(self.y1 + dy, 0), max(ih - 1 - h, 0))
        self.x1, self.y1, self.x2, self.y2 = nx1, ny1, nx1 + w, ny1 + h

# ---------- app ----------
class LabelerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fast Label v1.0")
        self.geometry(f"{LEFT_PANEL_WIDTH + IMAGE_AREA_WIDTH + 48}x{IMAGE_AREA_HEIGHT + 120}")
        self.minsize(LEFT_PANEL_WIDTH + IMAGE_AREA_WIDTH + 48, IMAGE_AREA_HEIGHT + 120)
        self.configure(bg=PALETTE["bg"])
        self.report_callback_exception = self._report_callback_exception
        self.protocol("WM_DELETE_WINDOW", self._wrap(self.on_close))
        ensure_dirs()
        self._init_style()

        # images
        self.image_paths: List[str] = []
        self.image_idx: int = -1
        self.image: Optional[Image.Image] = None

        # project index: per-image stats for navigator
        self.project_index: Dict[str, Dict[str, object]] = {}

        # cached display for performance
        self._cached_disp_size: Optional[Tuple[int,int]] = None
        self._cached_photo: Optional[ImageTk.PhotoImage] = None
        self._cached_pil: Optional[Image.Image] = None
        self._image_item: Optional[int] = None

        # ---
        self.dirty = False
        # memory
        self.annotations: Dict[str, List[Tuple[int,int,int,int,int]]] = {}

        # view transform
        self.base_scale: float = 1.0
        self.zoom: float = 1.0
        self.scale: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.min_zoom: float = 0.1
        self.max_zoom: float = 16.0

        # boxes
        self.boxes: List[Box] = []

        # interaction state
        self.dragging = False
        self.shift_held = False
        self.control_held = False
        self.alt_held = False                    # Alt = square/ratio constraint
        self.drag_start: Tuple[int,int] = (0,0)
        self.rubber_id: Optional[int] = None

        self.moving = False
        self.move_idx: Optional[int] = None
        self.move_start_img: Tuple[int,int] = (0,0)
        self.move_start_box: Optional[Tuple[int,int,int,int]] = None

        # multi-select group move
        self.move_selected_indices: List[int] = []
        self.move_start_boxes: Dict[int, Tuple[int,int,int,int]] = {}

        self.resizing = False
        self.resize_idx: Optional[int] = None
        self.resize_handle: Optional[str] = None
        self.resize_start_box: Optional[Tuple[int,int,int,int]] = None

        # ctrl-drag panning
        self.panning = False
        self.pan_start_canvas: Tuple[int,int] = (0,0)
        self.pan_start_offset: Tuple[float,float] = (0.0,0.0)

        # overlays
        self.crosshair_on = tk.BooleanVar(value=False)
        self.mouse_canvas_xy: Tuple[int,int] = (0,0)
        self.cursor_hidden = False

        # marquee (SHIFT + drag) — initialize to avoid AttributeError
        self.marquee_selecting = False
        self.marquee_id = None
        self.marquee_start: Tuple[int, int] = (0, 0)

        # clipboard
        self.copied_box: Optional[object] = None   # can be tuple or list of tuples
        self.paste_nudge = 8
        self.paste_count = 0

        # yolo
        self._yolo_prefilled = False
        self._prefill_running = False
        self._scan_all_running = False
        self._yolo_model_cache = {}

        # classes: start EMPTY (0 labels allowed)
        self.classes: Dict[int, Dict] = {}
        self.var_new_cls = tk.IntVar(value=0)  # will be set when a label exists

        # history (undo/redo) — stacks of snapshots
        self.undo_stack: List[Dict] = []
        self.redo_stack: List[Dict] = []

        # context menu
        self.ctx: Optional[tk.Menu] = None

        # ---------- layout ----------
        # Header
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text="Image Labeler", style="Header.TLabel").pack(side=tk.LEFT, padx=16, pady=12)

        # fixed-width header status so text never changes canvas width
        self.header_info = ttk.Label(
            header, text="Ready", style="HeaderInfo.TLabel",
            width=HEADER_STATUS_CHARS, anchor="e", justify="right")
        self.header_info.pack(side=tk.RIGHT, padx=16)

        # Content: wrap left + right in a single row to avoid middle gaps
        content = ttk.Frame(self, style="Right.TFrame")
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left sidebar
        left_container = ttk.Frame(content, style="Sidebar.TFrame", width=LEFT_PANEL_WIDTH)
        left_container.pack(side=tk.LEFT, fill=tk.Y)
        left_container.pack_propagate(False)
        left_scroll = ScrollableSidebar(left_container, width=LEFT_PANEL_WIDTH)
        left_scroll.pack(fill=tk.BOTH, expand=True)
        self.left_parent = left_scroll.inner

        # Left cards
        self._card_images(self.left_parent)
        self._card_project(self.left_parent)        # Project Navigator
        self._card_view(self.left_parent)
        self._card_yolo(self.left_parent)
        self.card_visibility = self._card_visibility(self.left_parent)
        self.card_newclass  = self._card_newclass(self.left_parent)
        self._card_class_manager(self.left_parent)  # Class Manager Pro
        self._card_actions(self.left_parent)

        # Right canvas area (fills remaining space)
        right = ttk.Frame(content, style="Right.TFrame")
        right.pack(side=tk.LEFT, padx=8, pady=8, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            right, bg=PALETTE["canvasbg"], highlightthickness=1,
            highlightbackground=PALETTE["outline2"])
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Floating zoom toolbar
        self.toolbar = ttk.Frame(right, style="Toolbar.TFrame")
        self.toolbar.place(x=10, y=10)
        ttk.Button(self.toolbar, text="−", style="Mini.TButton",
                   command=self._wrap(lambda: self.zoom_step(1/1.15, anchor="center"))).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(self.toolbar, text="Fit", style="Mini.TButton",
                   command=self._wrap(self.fit_to_screen)).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(self.toolbar, text="+", style="Mini.TButton",
                   command=self._wrap(lambda: self.zoom_step(1.15, anchor="center"))).pack(side=tk.LEFT)

        # Bindings
        self.canvas.bind("<ButtonPress-1>",   self._wrap(self.on_press))
        self.canvas.bind("<B1-Motion>",       self._wrap(self.on_drag))
        self.canvas.bind("<ButtonRelease-1>", self._wrap(self.on_release))
        self.canvas.bind("<Button-3>",        self._wrap(self.on_right_click))
        self.canvas.bind("<Motion>",          self._wrap(self.on_mouse_move))
        self.canvas.bind("<Enter>",           self._wrap(self.on_canvas_enter))
        self.canvas.bind("<Leave>",           self._wrap(self.on_canvas_leave))

        # Modifiers
        for k in ("<KeyPress-Control_L>", "<KeyPress-Control_R>"):
            self.bind(k, self._wrap(self.on_ctrl_down))
        for k in ("<KeyRelease-Control_L>", "<KeyRelease-Control_R>"):
            self.bind(k, self._wrap(self.on_ctrl_up))

        for k in ("<KeyPress-Shift_L>", "<KeyPress-Shift_R>"):
            self.bind(k, self._wrap(self.on_shift_down))
        for k in ("<KeyRelease-Shift_L>", "<KeyRelease-Shift_R>"):
            self.bind(k, self._wrap(self.on_shift_up))

        # ✅ Alt only (cross-platform). Do NOT bind Option_* — that crashes on Windows.
        for k in ("<KeyPress-Alt_L>", "<KeyPress-Alt_R>"):
            self.bind(k, self._wrap(self.on_alt_down))
        for k in ("<KeyRelease-Alt_L>", "<KeyRelease-Alt_R>"):
            self.bind(k, self._wrap(self.on_alt_up))

        # Shortcuts
        self.bind("<Delete>", self._wrap(self.on_delete_selected))
        self.bind("<KeyPress-Left>",  self._wrap(lambda e: self.nudge_selected(-1, 0)))
        self.bind("<KeyPress-Right>", self._wrap(lambda e: self.nudge_selected( 1, 0)))
        self.bind("<KeyPress-Up>",    self._wrap(lambda e: self.nudge_selected( 0,-1)))
        self.bind("<KeyPress-Down>",  self._wrap(lambda e: self.nudge_selected( 0, 1)))
        self.bind("<Control-c>", self._wrap(lambda e: self.copy_selected()))
        self.bind("<Control-C>", self._wrap(lambda e: self.copy_selected()))
        self.bind("<Control-v>", self._wrap(lambda e: self.paste_copied()))
        self.bind("<Control-V>", self._wrap(lambda e: self.paste_copied()))
        self.bind("<Control-s>", self._wrap(lambda e: self.on_save()))
        self.bind("<Control-S>", self._wrap(lambda e: self.on_save()))
        # Undo / Redo
        self.bind("<Control-z>", self._wrap(lambda e: self.on_undo()))
        self.bind("<Control-Z>", self._wrap(lambda e: self.on_undo()))
        self.bind("<Control-y>", self._wrap(lambda e: self.on_redo()))
        self.bind("<Control-Y>", self._wrap(lambda e: self.on_redo()))
        self.bind("<Control-Shift-Z>", self._wrap(lambda e: self.on_redo()))
        # Number hotkeys 1..9 -> select class by order
        for d in range(1,10):
            self.bind(str(d), self._wrap(lambda e, digit=d: self._on_number_hotkey(digit)))
            self.bind(f"<KP_{d}>", self._wrap(lambda e, digit=d: self._on_number_hotkey(digit)))

        # Zoom (Ctrl + Wheel)
        self.canvas.bind("<Control-MouseWheel>", self._wrap(self.on_ctrl_wheel))                # Windows/macOS
        self.canvas.bind("<Control-Button-4>",   self._wrap(lambda e: self.on_ctrl_wheel_linux(e, +1)))  # Linux up
        self.canvas.bind("<Control-Button-5>",   self._wrap(lambda e: self.on_ctrl_wheel_linux(e, -1)))  # Linux down

        # PAN via wheel / two-finger (vertical) and Shift+wheel (horizontal)
        self.canvas.bind("<MouseWheel>",          self._wrap(self.on_pan_wheel))               # vertical pan
        self.canvas.bind("<Shift-MouseWheel>",    self._wrap(self.on_pan_wheel_h))             # horizontal pan (Win/macOS)
        # Linux vertical pan:
        self.canvas.bind("<Button-4>",            self._wrap(lambda e: self.on_pan_wheel_linux(e, +1)))
        self.canvas.bind("<Button-5>",            self._wrap(lambda e: self.on_pan_wheel_linux(e, -1)))
        # Linux horizontal pan via Shift+Button-4/5:
        self.canvas.bind("<Shift-Button-4>",      self._wrap(lambda e: self.on_pan_wheel_linux_h(e, +1)))
        self.canvas.bind("<Shift-Button-5>",      self._wrap(lambda e: self.on_pan_wheel_linux_h(e, -1)))

        self._rebuild_context_menu()

        # Status bar (optional)
        self.status = tk.StringVar(value="Ready")
        if SHOW_STATUS_BAR:
            sb = ttk.Frame(self, style="Status.TFrame")
            sb.pack(side=tk.BOTTOM, fill=tk.X)
            ttk.Label(sb, textvariable=self.status, style="Status.TLabel").pack(side=tk.LEFT, padx=12, pady=6)

    # ---------- style ----------
    def _init_style(self):
        style = ttk.Style()
        try: style.theme_use("clam")
        except Exception: pass
        self.configure(bg=PALETTE["bg"])
        style.configure(".", background=PALETTE["bg"], foreground=PALETTE["fg"])
        style.configure("Header.TFrame", background=PALETTE["panel"])
        style.configure("Sidebar.TFrame", background=PALETTE["panel"])
        style.configure("Right.TFrame", background=PALETTE["bg"])
        style.configure("Card.TLabelframe", background=PALETTE["card"], bordercolor=PALETTE["outline"], relief="solid")
        style.configure("Card.TLabelframe.Label", background=PALETTE["card"], foreground=PALETTE["muted"], font=("Segoe UI", 10, "bold"))
        style.configure("Status.TFrame", background=PALETTE["panel"])
        style.configure("Status.TLabel", background=PALETTE["panel"], foreground=PALETTE["muted"], font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=PALETTE["panel"], foreground=PALETTE["fg"], font=("Segoe UI", 12, "bold"))
        style.configure("HeaderInfo.TLabel", background=PALETTE["panel"], foreground=PALETTE["muted"], font=("Segoe UI", 10))
        style.configure(
            "White.TLabel",
            background=PALETTE["card"],
            foreground="#ffffff"
        )

        style.configure("TButton",
                        background=PALETTE["card"], foreground=PALETTE["fg"],
                        bordercolor=PALETTE["outline"], focusthickness=1, focuscolor=PALETTE["accent"],
                        padding=(10,6))
        style.map("TButton",
                  background=[("active", PALETTE["hover"]), ("pressed", "#1a2430")],
                  foreground=[("active", PALETTE["fg"]),   ("pressed", PALETTE["fg"])])

        style.configure("Accent.TButton",
                        background=PALETTE["accent"], foreground="#0b0f12",
                        bordercolor="#2b6b67", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton",
                  background=[("active", "#59d9ce"), ("pressed", "#40bfb4")],
                  foreground=[("active", "#0b0f12"), ("pressed", "#0b0f12")])

        style.configure("Danger.TButton",
                        background=PALETTE["danger"], foreground="#0b0f12",
                        bordercolor="#7b2f2f", font=("Segoe UI", 10, "bold"))
        style.map("Danger.TButton",
                  background=[("active", "#ff7f7f"), ("pressed", "#e35f5f")],
                  foreground=[("active", "#0b0f12"), ("pressed", "#0b0f12")])

        style.configure("Toolbar.TFrame", background=PALETTE["card"])
        style.configure("Mini.TButton",
                        background=PALETTE["card"], foreground=PALETTE["fg"],
                        bordercolor=PALETTE["outline"], padding=(6,2), font=("Segoe UI", 9))
        style.map("Mini.TButton",
                  background=[("active", PALETTE["hover"])],
                  foreground=[("active", PALETTE["fg"])])
        style.configure("Dialog.TFrame", background=PALETTE["card"])
        style.configure("DialogHeading.TLabel",
                        background=PALETTE["card"], foreground=PALETTE["fg"],
                        font=("Segoe UI", 11, "bold"))
        style.configure("DialogText.TLabel",
                        background=PALETTE["card"], foreground=PALETTE["muted"])

        style.configure("ToggleOn.TButton", background="#89ffa6", foreground="#0b0f12")
        style.map("ToggleOn.TButton", background=[("active", "#66f18a")])

        style.configure("TLabel", background=PALETTE["card"], foreground=PALETTE["canvasbg"])
        style.configure("Muted.TLabel", background=PALETTE["card"], foreground=PALETTE["fg"])
        style.configure("TEntry", fieldbackground=PALETTE["panel"], foreground=PALETTE["fg"])
        style.configure("TCheckbutton", background=PALETTE["card"], foreground=PALETTE["fg"])
        style.configure("TRadiobutton", background=PALETTE["card"], foreground=PALETTE["fg"])
        style.map("TCheckbutton", background=[("active", PALETTE["hover"])], foreground=[("active", PALETTE["fg"])])
        style.map("TRadiobutton", background=[("active", PALETTE["hover"])], foreground=[("active", PALETTE["fg"])])
        style.configure("Sidebar.Vertical.TScrollbar",
                        background=PALETTE["outline2"], troughcolor=PALETTE["panel"],
                        bordercolor=PALETTE["outline"], lightcolor=PALETTE["outline2"],
                        darkcolor=PALETTE["outline"], arrowsize=14)
        style.map("Sidebar.Vertical.TScrollbar", background=[("active", PALETTE["hover"])])
        # Compact vertical scrollbar just for the PROJECT tree
        style.configure(
            "Project.Vertical.TScrollbar",
            background=PALETTE["outline2"],
            troughcolor=PALETTE["card"],
            bordercolor=PALETTE["outline"],
            lightcolor=PALETTE["outline2"],
            darkcolor=PALETTE["outline"],
            arrowsize=12,  # small arrows
        )
        style.map("Project.Vertical.TScrollbar", background=[("active", PALETTE["hover"])])

        style.configure("Treeview", background=PALETTE["card"], fieldbackground=PALETTE["card"], foreground=PALETTE["fg"])
        style.configure("TCombobox", fieldbackground=PALETTE["panel"], background=PALETTE["panel"])
        style.configure(
            "Filter.TCombobox",
            fieldbackground="#2b3542",  # input area background
            background="#2b3542",       # outer frame bg
            foreground="#ffffff"        # text color
        )
        style.map(
            "Filter.TCombobox",
            fieldbackground=[("readonly", "#2b3542"), ("!disabled", "#2b3542")],
            foreground=[("readonly", "#ffffff")],
            arrowcolor=[("readonly", "#ffffff"), ("!disabled", "#ffffff")],
            bordercolor=[("focus", PALETTE["accent"])]
        )

        # (Optional) style the dropdown list itself (affects all Combobox popups)
        self.option_add('*TCombobox*Listbox.background', '#222b36')
        self.option_add('*TCombobox*Listbox.foreground', '#ffffff')
        self.option_add('*TCombobox*Listbox.selectBackground', PALETTE['accent'])
        self.option_add('*TCombobox*Listbox.selectForeground', '#0b0f12')
        style.configure(
            "Dirty.TLabel",
            background=PALETTE["panel"],
            foreground="#ff3b3b",
            font=("Segoe UI", 16, "bold")
        )
    # ---------- UI cards ----------
    def _card_images(self, parent):
        card = ttk.Labelframe(parent, text="DATASET", style="Card.TLabelframe", padding=10)
        card.pack(fill=tk.X, pady=(8,6), padx=8)
        ttk.Button(card, text="📂  Open Images…", style="Accent.TButton",
                   command=self._wrap(self.open_images)).pack(fill=tk.X)
        row = ttk.Frame(card, style="Card.TLabelframe"); row.pack(fill=tk.X, pady=(8,0))
        ttk.Button(row, text="⟨  Prev", command=self._wrap(self.prev_image)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Next  ⟩", command=self._wrap(self.next_image)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.lbl_counts = ttk.Label(card, text="File: —\nBoxes: 0 (Visible: 0)", style="Muted.TLabel", justify="left")
        self.lbl_counts.pack(anchor="w", pady=(8,0))

    def _card_project(self, parent):
        card = ttk.Labelframe(parent, text="PROJECT", style="Card.TLabelframe", padding=10)
        card.pack(fill=tk.BOTH, pady=6, padx=8)

        # Filters
        frow = ttk.Frame(card, style="Card.TLabelframe"); frow.pack(fill=tk.X)
        ttk.Label(frow, text="Filter:", style="Muted.TLabel").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar(value="All")
        self.filter_combo = ttk.Combobox(
                frow,
                textvariable=self.filter_var,
                state="readonly",
                values=["All","Labeled","Unlabeled"],
                style="Filter.TCombobox"   # <- add this
            )
        self.filter_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8,8))
        self.filter_combo.bind("<<ComboboxSelected>>", self._wrap(lambda e: self._rebuild_project_tree()))
        ttk.Button(frow, text="Refresh", command=self._wrap(self._refresh_project_index)).pack(side=tk.LEFT)

        # Tree
        tree_frame = ttk.Frame(card, style="Card.TLabelframe")
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(6,0))

        columns = ("boxes",)
        # Bind wheel/trackpad to the PROJECT tree itself

        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="tree headings",
            selectmode="browse",
            height=8
        )
        self.tree.heading("#0", text="File")
        self.tree.heading("boxes", text="Boxes")
        self.tree.column("#0", stretch=True, width=220)
        self.tree.column("boxes", stretch=False, width=60, anchor="center")

        # 🔽 Small, styled vertical scrollbar
        vsb = ttk.Scrollbar(
            tree_frame,
            orient="vertical",
            command=self.tree.yview,
            style="Project.Vertical.TScrollbar",
        )
        self.tree.bind("<MouseWheel>", self._wrap(self._tree_on_mousewheel))  # Win/macOS
        self.tree.bind("<Button-4>",   self._wrap(lambda e: self._tree_on_mousewheel_linux(e, +1)))  # Linux up
        self.tree.bind("<Button-5>",   self._wrap(lambda e: self._tree_on_mousewheel_linux(e, -1)))  # Linux down
        self.tree.configure(yscrollcommand=vsb.set)

        # Layout: tree left, scrollbar right
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Bindings
        self.tree.bind("<Double-1>", self._wrap(self._on_tree_double_click))
        self.tree.bind("<Return>",   self._wrap(self._on_tree_enter))

        # "Has: Class X" filters populated dynamically
        self._update_filter_with_classes()

    def _card_view(self, parent):
        card = ttk.Labelframe(parent, text="VIEW", style="Card.TLabelframe", padding=10)
        card.pack(fill=tk.X, pady=6, padx=8)
        def toggle_cross():
            self.crosshair_on.set(self.crosshair_on.get() if False else not self.crosshair_on.get())
            self._set_status(f"Cross-hair {'ON' if self.crosshair_on.get() else 'OFF'}")
            self.redraw()
        self.cross_btn = ttk.Button(card, text="✚  Cross-hair OFF", command=self._wrap(toggle_cross))
        self.cross_btn.pack(fill=tk.X)
    def _tree_on_mousewheel(self, e):
        """
        Scroll the PROJECT tree when hovered, but only if it can scroll further.
        If at bounds, let the event bubble to the sidebar (no 'break').
        """
        try:
            # Tk uses +120/-120 steps; match your sidebar logic
            units = -1 if e.delta > 0 else 1
            first, last = self.tree.yview()  # fractions [0..1] of the visible region
            at_top    = first <= 0.0
            at_bottom = last  >= 1.0

            # If we can scroll in this direction, do it and consume the event.
            if (units < 0 and not at_top) or (units > 0 and not at_bottom):
                self.tree.yview_scroll(units, "units")
                return "break"  # stop propagation -> sidebar won’t scroll
            # Else: at edge; do nothing so the sidebar router can handle it
        except Exception:
            pass  # fall through to sidebar if anything odd happens

    def _tree_on_mousewheel_linux(self, e, direction: int):
        """
        Linux Button-4/5 variant (+1 up, -1 down).
        Same edge-aware behavior as _tree_on_mousewheel.
        """
        try:
            units = -1 if direction > 0 else 1  # keep same sign convention
            first, last = self.tree.yview()
            at_top    = first <= 0.0
            at_bottom = last  >= 1.0
            if (units < 0 and not at_top) or (units > 0 and not at_bottom):
                self.tree.yview_scroll(units, "units")
                return "break"
        except Exception:
            pass

    def _card_yolo(self, parent):
        card = ttk.Labelframe(parent, text="YOLO PREFILL", style="Card.TLabelframe", padding=10)
        card.pack(fill=tk.X, pady=6, padx=8)

        self.var_prefill = tk.BooleanVar(value=bool(_YOLO_OK))
        ttk.Checkbutton(card, text="Enable prefill", variable=self.var_prefill).pack(anchor="w")

        row = ttk.Frame(card, style="Card.TLabelframe"); row.pack(fill=tk.X, pady=(6,0))
        ttk.Label(row, text="Model path:", style="Muted.TLabel").pack(side=tk.LEFT)
        self.var_model = tk.StringVar(value=DEFAULT_YOLO_MODEL)
        ent = ttk.Entry(row, textvariable=self.var_model)
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8,8))
        ttk.Button(row, text="Browse…", command=self._wrap(self.browse_model_file)).pack(side=tk.LEFT)

        row2 = ttk.Frame(card, style="Card.TLabelframe"); row2.pack(fill=tk.X, pady=(8,0))
        ttk.Button(row2, text="🔮  Prefill (once)", style="Accent.TButton",
                   command=self._wrap(self.on_prefill_once)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row2, text="🧭  Scan All", command=self._wrap(self.on_scan_all)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8,0))

    def _card_visibility(self, parent):
        card = ttk.Labelframe(parent, text="VISIBILITY", style="Card.TLabelframe", padding=10)
        card.pack(fill=tk.X, pady=6, padx=8)
        self.visibility_container = ttk.Frame(card, style="Card.TLabelframe")
        self.visibility_container.pack(fill=tk.X)
        self._rebuild_visibility_ui()
        return card

    def _card_newclass(self, parent):
        card = ttk.Labelframe(parent, text="CLASSES", style="Card.TLabelframe", padding=10)
        card.pack(fill=tk.X, pady=6, padx=8)
        self.newclass_container = ttk.Frame(card, style="Card.TLabelframe")
        self.newclass_container.pack(fill=tk.X)

        row = ttk.Frame(card, style="Card.TLabelframe"); row.pack(fill=tk.X, pady=(8,0))
        ttk.Button(row, text="➕  Add label…", style="Accent.TButton",
                   command=self._wrap(self._add_label_dialog)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="➖  Remove label…", style="Danger.TButton",
                   command=self._wrap(self._remove_label_dialog)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self._rebuild_newclass_ui()
        return card

    def _card_class_manager(self, parent):
        card = ttk.Labelframe(parent, text="CLASS MANAGER PRO", style="Card.TLabelframe", padding=10)
        card.pack(fill=tk.X, pady=6, padx=8)
        row = ttk.Frame(card, style="Card.TLabelframe"); row.pack(fill=tk.X)
        ttk.Button(row, text="✏️  Rename…", command=self._wrap(self._rename_label_dialog)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="🎨  Color…", command=self._wrap(self._color_label_dialog)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row, text="🔀  Merge…", command=self._wrap(self._merge_labels_dialog)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(card, text="Hotkeys: 1→class id 0 (Locked), 2→class id 1 (Unlocked), etc.",
                  style="Muted.TLabel").pack(anchor="w", pady=(6,0))

    def _card_actions(self, parent):
        card = ttk.Labelframe(parent, text="ACTIONS", style="Card.TLabelframe", padding=10)
        card.pack(fill=tk.X, pady=6, padx=8)
        ttk.Button(card, text="🧹  Clear Boxes", style="Danger.TButton",
                   command=self._wrap(self.clear_boxes)).pack(fill=tk.X)
        ttk.Button(card, text="💾  Save labels", style="Accent.TButton",
                   command=self._wrap(self.on_save)).pack(fill=tk.X, pady=(8,0))


    # ---------- Project Navigator helpers ----------
    def _update_filter_with_classes(self):
        vals = ["All","Labeled","Unlabeled"]
        for cid in sorted(self.classes):
            nm = self.classes[cid]["name"]
            vals.append(f"Has: {nm} ({cid})")
        cur = getattr(self, "filter_var", None)
        if cur is None:
            return
        cur_val = self.filter_var.get()
        self.filter_combo["values"] = vals
        if cur_val not in vals:
            self.filter_var.set("All")
    def _ask_save_on_close(self) -> str:
        """
        Returns one of: 'save', 'dont', 'cancel'
        """
        dlg = tk.Toplevel(self)
        dlg.title("Unsaved changes")
        dlg.configure(bg=PALETTE["card"])
        dlg.transient(self)
        dlg.grab_set()

        # center dialog over parent
        self.update_idletasks()
        px = self.winfo_rootx()
        py = self.winfo_rooty()
        pw = self.winfo_width()
        ph = self.winfo_height()
        dlg.update_idletasks()
        w, h = 360, 140
        x = px + (pw - w)//2
        y = py + (ph - h)//2
        dlg.geometry(f"{w}x{h}+{x}+{y}")

        # content
        frm = ttk.Frame(dlg, padding=16, style="Card.TLabelframe")
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Save changes before closing?", style="White.TLabel").pack(anchor="w")

        choice = {"val": "cancel"}  # default if the window is closed

        btns = ttk.Frame(frm, style="Card.TLabelframe")
        btns.pack(side=tk.BOTTOM, anchor="e", pady=(16,0))

        def _set(v):
            choice["val"] = v
            dlg.destroy()

        ttk.Button(btns, text="Save",       command=lambda: _set("save"),
                   style="Accent.TButton").pack(side=tk.RIGHT, padx=(8,0))
        ttk.Button(btns, text="Don’t Save", command=lambda: _set("dont")).pack(side=tk.RIGHT, padx=(8,0))
        ttk.Button(btns, text="Cancel",     command=lambda: _set("cancel")).pack(side=tk.RIGHT)

        dlg.protocol("WM_DELETE_WINDOW", lambda: _set("cancel"))
        dlg.wait_window()
        return choice["val"]
    def _modal_choice(self, title: str, message: str, buttons, width: int = 480):
        """
        Reusable modal with right-aligned buttons.
        buttons: list[(text, ttk_style, return_value)] in LEFT→RIGHT visual order.
                 The FIRST item is treated as the primary (Enter).
        Returns chosen return_value (or None on cancel/close).
        """
        dlg = tk.Toplevel(self)
        dlg.withdraw()                      # avoids initial flicker
        dlg.title(title)
        dlg.transient(self)
        dlg.configure(bg=PALETTE["card"])
        dlg.resizable(False, False)

        result = {"val": None}
        def choose(val):
            result["val"] = val
            try: dlg.grab_release()
            except Exception: pass
            dlg.destroy()

        dlg.protocol("WM_DELETE_WINDOW", lambda: choose(None))
        dlg.bind("<Escape>", lambda e: choose(None))

        # ---- layout ------------------------------------------------------------
        outer = ttk.Frame(dlg, style="Dialog.TFrame", padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer, style="Dialog.TFrame")
        top.pack(fill=tk.X)

        # icon + text
        ttk.Label(top, text="⚠️", style="DialogHeading.TLabel").pack(side=tk.LEFT, padx=(2, 10))
        text_box = ttk.Frame(top, style="Dialog.TFrame"); text_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(text_box, text=title, style="DialogHeading.TLabel").pack(anchor="w")
        ttk.Label(text_box, text=message, style="DialogText.TLabel",
                  wraplength=width-120, justify="left").pack(anchor="w", pady=(4, 0))

        # buttons (right-aligned with a spacer column)
        btn_row = ttk.Frame(outer, style="Dialog.TFrame"); btn_row.pack(fill=tk.X, pady=(12, 0))
        btn_row.grid_columnconfigure(0, weight=1)

        primary_btn = None
        for i, (text, style_name, ret) in enumerate(buttons, start=1):
            b = ttk.Button(btn_row, text=text, style=style_name, command=lambda v=ret: choose(v))
            b.grid(row=0, column=i, padx=(8 if i > 1 else 0, 0))
            if primary_btn is None:
                primary_btn = b

        dlg.bind("<Return>", lambda e: primary_btn.invoke())

        # ---- show centered & modal --------------------------------------------
        dlg.update_idletasks()
        dlg.deiconify()
        w = max(width, outer.winfo_reqwidth() + 28)
        h = outer.winfo_reqheight() + 28
        x = self.winfo_rootx() + (self.winfo_width() - w)//2
        y = self.winfo_rooty() + (self.winfo_height() - h)//2
        dlg.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")

        dlg.grab_set()
        primary_btn.focus_set()
        dlg.wait_window()
        return result["val"]

    
    def _refresh_project_index(self):
        self._rebuild_project_index()
        self._rebuild_project_tree()
        self._set_status("Project refreshed.")

    def _rebuild_project_index(self):
        self.project_index.clear()
        for p in self.image_paths:
            boxes_count = 0
            classes_set: Set[int] = set()
            if p in self.annotations:
                snap = self.annotations[p]
                boxes_count = len(snap)
                for *_coords, cls in snap:
                    classes_set.add(int(cls))
            else:
                try:
                    txt = self._yolo_txt_path_for(p)
                    if os.path.exists(txt):
                        iw, ih = Image.open(p).size
                        snap = self._read_yolo_txt_for_path(p, (iw, ih))
                        boxes_count = len(snap)
                        for *_c, cls in snap:
                            classes_set.add(int(cls))
                except Exception as ex:
                    log_exc("_rebuild_project_index", ex)
            self.project_index[p] = {"boxes": boxes_count, "classes": classes_set}

    def _rebuild_project_tree(self):
        if not hasattr(self, "tree"):
            return
        for i in self.tree.get_children():
            self.tree.delete(i)
        # Rebuild filter with latest classes
        self._update_filter_with_classes()

        flt = self.filter_var.get()
        match_cid: Optional[int] = None
        if flt.startswith("Has:"):
            try:
                tail = flt.split("(")[-1]
                match_cid = int(tail[:-1])  # drop ')'
            except Exception:
                match_cid = None

        for p in self.image_paths:
            info = self.project_index.get(p, {"boxes":0,"classes":set()})
            b = int(info.get("boxes", 0))
            cset = set(info.get("classes", set()))
            if flt == "Labeled" and b == 0:
                continue
            if flt == "Unlabeled" and b > 0:
                continue
            if match_cid is not None and match_cid not in cset:
                continue
            iid = p  # use full path as item id
            self.tree.insert("", "end", iid=iid, text=os.path.basename(p), values=(b,))

    def _on_tree_double_click(self, _evt=None):
        sel = self.tree.selection()
        if not sel: return
        path = sel[0]
        if path in self.image_paths:
            self._autosave_current(silent=True)
            self.image_idx = self.image_paths.index(path)
            self._history_reset()
            self._load_current_image()

    def _on_tree_enter(self, _evt=None):
        self._on_tree_double_click()

    # ---------- dynamic class UI rebuild ----------
    def _rebuild_visibility_ui(self):
        if hasattr(self, "visibility_container"):
            for w in self.visibility_container.winfo_children():
                w.destroy()
        if not self.classes and hasattr(self, "visibility_container"):
            ttk.Label(self.visibility_container, text="No labels yet.", style="Muted.TLabel").pack(anchor="w")
            return
        if hasattr(self, "visibility_container"):
            for cid in sorted(self.classes):
                info = self.classes[cid]
                var = info.get("show") or tk.BooleanVar(value=True)
                info["show"] = var
                text = f"{info['name']} ({cid})"
                ttk.Checkbutton(self.visibility_container, text=text, variable=var,
                                command=self._wrap(self.redraw)).pack(anchor="w")

    def _rebuild_newclass_ui(self):
        if hasattr(self, "newclass_container"):
            for w in self.newclass_container.winfo_children():
                w.destroy()
        if not self.classes and hasattr(self, "newclass_container"):
            ttk.Label(self.newclass_container, text="No labels. Add one or run YOLO Prefill.", style="Muted.TLabel").pack(anchor="w")
            self.var_new_cls.set(0)
            self._rebuild_context_menu()
            self._update_filter_with_classes()
            return
        if hasattr(self, "newclass_container"):
            for cid in sorted(self.classes):
                info = self.classes[cid]
                text = f"{info['name']} ({cid})"
                ttk.Radiobutton(self.newclass_container, text=text, variable=self.var_new_cls, value=cid).pack(anchor="w")
        if self.classes:
            if self.var_new_cls.get() not in self.classes:
                self.var_new_cls.set(sorted(self.classes)[0])
        self._rebuild_context_menu()
        self._update_filter_with_classes()

    def _rebuild_context_menu(self):
        if self.ctx is None:
            self.ctx = tk.Menu(self, tearoff=0, bg=PALETTE["card"], fg=PALETTE["fg"],
                               activebackground=PALETTE["hover"], activeforeground=PALETTE["fg"],
                               relief="flat", borderwidth=1)
        else:
            self.ctx.delete(0, tk.END)
        if self.classes:
            for cid in sorted(self.classes):
                nm = self.classes[cid]["name"]
                self.ctx.add_command(label=f"Set to {nm} ({cid})", command=lambda c=cid: self.set_selected_class(c))
            self.ctx.add_separator()
        self.ctx.add_command(label="Delete Selected", command=lambda: self.on_delete_selected())

    # ---------- class add/remove/manager helpers ----------
    def _next_free_id(self) -> int:
        cid = 0
        used = set(self.classes.keys())
        while cid in used: cid += 1
        return cid

    def _auto_color_for(self, cid: int) -> str:
        if cid == YOLO_UNLOCKED_ID: return PALETTE["canvasbg"]   # per your change
        if cid == YOLO_LOCKED_ID:   return PALETTE["danger"]
        idx = (cid - 2) % len(EXTRA_COLORS)
        return EXTRA_COLORS[idx]

    def _add_label_dialog(self):
        name = simpledialog.askstring("Add Label", "Class name:", parent=self)
        if not name: return
        if any(info["name"].lower() == name.lower() for info in self.classes.values()):
            messagebox.showwarning("Duplicate", f'Label "{name}" already exists.')
            return
        self._push_undo()
        cid = self._next_free_id()
        self.classes[cid] = {"name": name, "color": self._auto_color_for(cid), "show": tk.BooleanVar(value=True)}
        if not self.classes or self.var_new_cls.get() not in self.classes:
            self.var_new_cls.set(cid)
        self._rebuild_visibility_ui()
        self._rebuild_newclass_ui()
        self.redraw()
        self._set_status(f'Added label "{name}" as id {cid}.')

    def _remove_label_dialog(self):
        if not self.classes:
            messagebox.showinfo("No labels", "There are no labels to remove.")
            return
        ids_sorted = sorted(self.classes)
        names = [f'{cid}: {self.classes[cid]["name"]}' for cid in ids_sorted]
        choice = simpledialog.askstring("Remove Label",
                                        "Enter class id to remove:\n" + "\n".join(names),
                                        parent=self)
        if choice is None: return
        try:
            rid = int(choice)
        except Exception:
            messagebox.showerror("Invalid", "Please enter a numeric class id.")
            return
        if rid not in self.classes:
            messagebox.showerror("Not found", f"Class id {rid} does not exist.")
            return
        count = sum(1 for b in self.boxes if b.cls == rid)
        if count > 0:
            if not messagebox.askyesno("Class in use",
                                       f'This class "{self.classes[rid]["name"]}" (id {rid}) has {count} boxes.\n'
                                       f"Delete ALL those boxes and remove the class?"):
                return
        self._push_undo()
        if count > 0:
            self.boxes = [b for b in self.boxes if b.cls != rid]
            self._remember_current()
        del self.classes[rid]
        if self.classes and self.var_new_cls.get() not in self.classes:
            self.var_new_cls.set(sorted(self.classes)[0])
        self._rebuild_visibility_ui()
        self._rebuild_newclass_ui()
        self.redraw()
        self._set_status(f"Removed class id {rid}.")

    # --- Class Manager Pro dialogs ---
    def _rename_label_dialog(self):
        if not self.classes:
            messagebox.showinfo("No labels", "There are no labels to rename.")
            return
        ids_sorted = sorted(self.classes)
        cid_str = simpledialog.askstring("Rename Label",
                                         "Enter class id to rename:\n" + "\n".join(f"{c}: {self.classes[c]['name']}" for c in ids_sorted),
                                         parent=self)
        if cid_str is None: return
        try:
            cid = int(cid_str)
        except Exception:
            messagebox.showerror("Invalid", "Please enter a numeric class id.")
            return
        if cid not in self.classes:
            messagebox.showerror("Not found", f"Class id {cid} does not exist.")
            return
        new_name = simpledialog.askstring("Rename Label", f'New name for id {cid} ("{self.classes[cid]["name"]}"):', parent=self)
        if not new_name: return
        if any(info["name"].lower()==new_name.lower() and c!=cid for c,info in self.classes.items()):
            messagebox.showwarning("Duplicate", f'Label "{new_name}" already exists.')
            return
        self._push_undo()
        self.classes[cid]["name"] = new_name
        self._rebuild_visibility_ui()
        self._rebuild_newclass_ui()
        self.redraw()
        self._set_status(f'Renamed class {cid} to "{new_name}".')

    def _color_label_dialog(self):
        if not self.classes:
            messagebox.showinfo("No labels", "There are no labels to color.")
            return
        cid_str = simpledialog.askstring("Class Color",
                                         "Enter class id to recolor:\n" + "\n".join(f"{c}: {self.classes[c]['name']}  [{self.classes[c]['color']}]"
                                                                                    for c in sorted(self.classes)),
                                         parent=self)
        if cid_str is None: return
        try:
            cid = int(cid_str)
        except Exception:
            messagebox.showerror("Invalid", "Please enter a numeric class id.")
            return
        if cid not in self.classes:
            messagebox.showerror("Not found", f"Class id {cid} does not exist.")
            return
        rgb, hexcolor = colorchooser.askcolor(color=self.classes[cid]["color"], title=f"Pick color for {self.classes[cid]['name']} ({cid})")
        if hexcolor:
            self._push_undo()
            self.classes[cid]["color"] = hexcolor
            self.redraw()
            self._set_status(f"Color set for class {cid}: {hexcolor}")

    def _merge_labels_dialog(self):
        if len(self.classes) < 2:
            messagebox.showinfo("Not enough labels", "Need at least two labels to merge.")
            return
        ids_sorted = sorted(self.classes)
        list_text = "\n".join(f"{c}: {self.classes[c]['name']}" for c in ids_sorted)
        src_str = simpledialog.askstring("Merge Labels — Source", "Move FROM class id:\n"+list_text, parent=self)
        if src_str is None: return
        dst_str = simpledialog.askstring("Merge Labels — Target", "Move INTO class id:\n"+list_text, parent=self)
        if dst_str is None: return
        try:
            src = int(src_str); dst = int(dst_str)
        except Exception:
            messagebox.showerror("Invalid", "Please enter numeric class ids.")
            return
        if src not in self.classes or dst not in self.classes:
            messagebox.showerror("Not found", "Source or target id does not exist.")
            return
        if src == dst:
            messagebox.showwarning("Same class", "Source and target are the same.")
            return
        if not messagebox.askyesno("Confirm Merge",
                                   f'Merge "{self.classes[src]["name"]}" (id {src}) into "{self.classes[dst]["name"]}" (id {dst})?\n'
                                   f"All boxes of id {src} will become id {dst}."):
            return
        self._push_undo()
        for b in self.boxes:
            if b.cls == src:
                b.cls = dst
        del self.classes[src]
        if self.var_new_cls.get() not in self.classes and self.classes:
            self.var_new_cls.set(sorted(self.classes)[0])
        self._remember_current()
        self._rebuild_visibility_ui()
        self._rebuild_newclass_ui()
        self.redraw()
        self._set_status(f"Merged class {src} into {dst}.")

    # ---------- status helpers ----------
    def _short_status(self, s: str, limit: int = STATUS_MAX_CHARS) -> str:
        if len(s) <= limit: return s
        head = s[: int(limit * 0.45)]
        tail = s[-int(limit * 0.45):]
        return f"{head} … {tail}"
    def _set_status(self, s: str):
        self.header_info.config(text=self._short_status(s, HEADER_STATUS_CHARS))
        if SHOW_STATUS_BAR:
            self.status.set(self._short_status(s))

    # ---------- safety ----------
    def _wrap(self, func):
        def inner(*a, **kw):
            try:
                return func(*a, **kw)
            except BaseException as ex:
                log_exc(func.__name__, ex)
        return inner

    def _report_callback_exception(self, exc, val, tb):
        log_exc("Tk callback", val)

        

    # ---------- files & nav ----------
    def open_images(self):
        self._autosave_current(silent=True)
        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("All files","*.*")]
        )
        if not paths: return
        self.image_paths = list(paths)
        self.image_idx = 0
        self._history_reset()
        self._rebuild_project_index()
        self._rebuild_project_tree()
        self._load_current_image()

    def prev_image(self):
        if not self.image_paths: return
        if self.image_idx <= 0:
            self._set_status("Start of list.")
            try: messagebox.showinfo("Info", "You are at the first image.")
            except Exception: pass
            return
        self._autosave_current(silent=True)
        self.image_idx -= 1
        self._history_reset()
        self._load_current_image()

    def next_image(self):
        if not self.image_paths: return
        if self.image_idx >= len(self.image_paths) - 1:
            self._set_status("Reached end of list.")
            try: messagebox.showinfo("Info", "You are at the last image.")
            except Exception: pass
            return
        self._autosave_current(silent=True)
        self.image_idx += 1
        self._history_reset()
        self._load_current_image()

    def _load_current_image(self):
        self.boxes.clear()
        self._yolo_prefilled = False
        self._prefill_running = False
        self._scan_all_running = False
        self.resizing = False; self.moving = False

        p = self.image_paths[self.image_idx]
        try:
            self.image = Image.open(p).convert("RGB")
        except Exception as ex:
            log_exc("open_image", ex)
            messagebox.showerror("Open image failed", f"{p}\n{ex}")
            return

        # Reset zoom to fit
        self.zoom = 1.0
        self._recenter_fit()
        self._clear_cache()  # new image => drop cache

        restored = False
        if p in self.annotations:
            self._restore_from_snapshot(self.annotations[p]); restored = True
        else:
            try:
                snap = self._read_yolo_txt_for_path(p, self.image.size)
                if snap:
                    for *_, cls in snap:
                        if cls not in self.classes:
                            self.classes[cls] = {"name": f"class_{cls}",
                                                 "color": self._auto_color_for(cls),
                                                 "show": tk.BooleanVar(value=True)}
                    self._rebuild_visibility_ui()
                    self._rebuild_newclass_ui()
                    self._restore_from_snapshot(snap); restored = True
            except Exception as ex:
                log_exc("load_labels", ex)

        # Update project index for current
        self._update_project_index_for_current()

        fn = os.path.basename(p)
        msg = f"Loaded: {fn}  [{self.image_idx+1}/{len(self.image_paths)}]"
        if restored: msg += "  (restored)"
        self._set_status(msg)
        self.dirty = False
        self.redraw()

    def on_close(self):
        # Only prompt if there are unsaved changes for the current image
        if getattr(self, "dirty", False):
            ans = self._ask_save_on_close()
            if ans == "cancel":
                return
            if ans == "save":
                ok = self._do_save(silent=True)
                if not ok:
                    # If save failed and user didn’t cancel, keep the app open
                    return
            # 'dont' falls through and closes without saving
        self.destroy()


    # ---------- snapshots & (auto)save ----------
    def _snapshot(self) -> List[Tuple[int,int,int,int,int]]:
        return [(b.x1, b.y1, b.x2, b.y2, b.cls) for b in self.boxes]
    def _remember_current(self):
        if not self.image_paths or self.image_idx < 0: return
        self.annotations[self.image_paths[self.image_idx]] = self._snapshot()
        self._update_project_index_for_current()  # keep navigator updated
        self.dirty = True

    def _restore_from_snapshot(self, snap: List[Tuple[int,int,int,int,int]]):
        self.boxes = []
        for t in snap:
            try:
                x1,y1,x2,y2,cls = t
                self.boxes.append(Box(x1,y1,x2,y2,cls,selected=False))
            except Exception as ex:
                log_exc("_restore_from_snapshot", ex)

    def _yolo_txt_path_for(self, img_path:str) -> str:
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(OUTPUT_LBL_DIR, base + ".txt")

    def _write_yolo_txt_for_path(self, img_path: str, boxes: List[Box]):
        if not img_path: return
        try:
            im = self.image if (self.image_paths and self.image_paths[self.image_idx] == img_path) else Image.open(img_path).convert("RGB")
        except Exception:
            im = self.image
        if im is None: return
        txt_path = self._yolo_txt_path_for(img_path)
        iw, ih = im.size
        lines = []
        for b in boxes:
            cx = (b.x1 + b.x2) / 2.0 / iw
            cy = (b.y1 + b.y2) / 2.0 / ih
            nw = (b.x2 - b.x1) / iw
            nh = (b.y2 - b.y1) / ih
            lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if self.image_paths and 0 <= self.image_idx < len(self.image_paths):
            if os.path.abspath(img_path) == os.path.abspath(self.image_paths[self.image_idx]):
                self.dirty = False

    def _read_yolo_txt_for_path(self, img_path: str, img_size: Tuple[int,int]) -> List[Tuple[int,int,int,int,int]]:
        txt_path = self._yolo_txt_path_for(img_path)
        if not os.path.exists(txt_path): return []
        iw, ih = img_size
        snap: List[Tuple[int,int,int,int,int]] = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) != 5: continue
                cls = int(float(parts[0]))
                cx = float(parts[1]) * iw
                cy = float(parts[2]) * ih
                w  = float(parts[3]) * iw
                h  = float(parts[4]) * ih
                x1 = int(round(cx - w/2)); y1 = int(round(cy - h/2))
                x2 = int(round(cx + w/2)); y2 = int(round(cy + h/2))
                x1 = max(0, min(iw-1, x1)); y1 = max(0, min(ih-1, y1))
                x2 = max(0, min(iw-1, x2)); y2 = max(0, min(ih-1, y2))
                if x2 <= x1 or y2 <= y1: continue
                snap.append((x1,y1,x2,y2,cls))
        return snap

    def _autosave_current(self, silent: bool):
        if not self.image_paths or self.image_idx < 0 or self.image is None: return
        img_path = self.image_paths[self.image_idx]
        try:
            self._remember_current()
            self._write_yolo_txt_for_path(img_path, self.boxes)
            if not silent and SHOW_STATUS_BAR:
                self._set_status(f"Autosaved: {os.path.basename(self._yolo_txt_path_for(img_path))}")
        except Exception as ex:
            log_exc("autosave", ex)
            if not silent:
                try: messagebox.showwarning("Autosave failed", str(ex))
                except Exception: pass

    def _update_project_index_for_current(self):
        if not self.image_paths or self.image_idx < 0: return
        p = self.image_paths[self.image_idx]
        boxes_count = len(self.boxes)
        classes_set = set(b.cls for b in self.boxes)
        self.project_index[p] = {"boxes": boxes_count, "classes": classes_set}
        self._rebuild_project_tree()

    # ---------- history (undo/redo) ----------
    def _history_reset(self):
        self.undo_stack.clear()
        self.redo_stack.clear()

    def _capture_snapshot_state(self) -> Dict:
        classes_plain: Dict[int, Dict] = {}
        for cid, info in self.classes.items():
            classes_plain[int(cid)] = {
                "name": str(info.get("name","")),
                "color": str(info.get("color", PALETTE["accent"])),
                "show": bool(info.get("show").get() if isinstance(info.get("show"), tk.BooleanVar) else True)
            }
        return {
            "boxes": self._snapshot(),
            "classes": classes_plain,
            "selected_cls": int(self.var_new_cls.get()) if self.classes else 0,
            "yolo_prefilled": bool(self._yolo_prefilled),
        }

    def _apply_snapshot_state(self, snap: Dict):
        cls_plain: Dict[int, Dict] = snap.get("classes", {})
        self.classes = {}
        for cid, info in cls_plain.items():
            self.classes[int(cid)] = {
                "name": info.get("name",""),
                "color": info.get("color", PALETTE["accent"]),
                "show": tk.BooleanVar(value=bool(info.get("show", True)))
            }
        self._restore_from_snapshot(snap.get("boxes", []))
        sel = snap.get("selected_cls", 0)
        if self.classes and sel in self.classes:
            self.var_new_cls.set(sel)
        elif self.classes:
            self.var_new_cls.set(sorted(self.classes)[0])
        else:
            self.var_new_cls.set(0)
        self._yolo_prefilled = bool(snap.get("yolo_prefilled", False))
        self._rebuild_visibility_ui()
        self._rebuild_newclass_ui()
        self._remember_current()
        self.redraw()

    def _push_undo(self):
        self.undo_stack.append(self._capture_snapshot_state())
        if len(self.undo_stack) > MAX_HISTORY:
            self.undo_stack = self.undo_stack[-MAX_HISTORY:]
        self.redo_stack.clear()

    def on_undo(self):
        if not self.undo_stack:
            self._set_status("Nothing to undo.")
            return
        current = self._capture_snapshot_state()
        last = self.undo_stack.pop()
        self.redo_stack.append(current)
        if len(self.redo_stack) > MAX_HISTORY:
            self.redo_stack = self.redo_stack[-MAX_HISTORY:]
        self._apply_snapshot_state(last)
        self._set_status("Undo")

    def on_redo(self):
        if not self.redo_stack:
            self._set_status("Nothing to redo.")
            return
        current = self._capture_snapshot_state()
        nxt = self.redo_stack.pop()
        self.undo_stack.append(current)
        if len(self.undo_stack) > MAX_HISTORY:
            self.undo_stack = self.undo_stack[-MAX_HISTORY:]
        self._apply_snapshot_state(nxt)
        self._set_status("Redo")

    # ---------- YOLO helpers/actions ----------
    def browse_model_file(self):
        path = filedialog.askopenfilename(
            title="Select YOLO model (best.pt)",
            filetypes=[("YOLO PyTorch", "*.pt"), ("All files", "*.*")]
        )
        if not path: return
        self.var_model.set(path)
        self._set_status(f"Model selected: {os.path.basename(path)}")

    def _get_yolo_model(self, path: str):
        key = os.path.abspath(path)
        mdl = self._yolo_model_cache.get(key)
        if mdl is not None: return mdl
        mdl = YOLO(key)
        self._yolo_model_cache[key] = mdl
        return mdl

    def _sanitize_and_clip(self, xyxy, iw, ih) -> Optional[Tuple[int,int,int,int]]:
        if xyxy is None or len(xyxy) != 4: return None
        x1, y1, x2, y2 = xyxy
        for v in (x1, y1, x2, y2):
            if v is None: return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
        x1 = int(max(0, min(iw - 1, round(x1))))
        y1 = int(max(0, min(ih - 1, round(y1))))
        x2 = int(max(0, min(iw - 1, round(x2))))
        y2 = int(max(0, min(ih - 1, round(y2))))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        if (x2 - x1) < MIN_SIDE or (y2 - y1) < MIN_SIDE: return None
        return x1, y1, x2, y2

    def _detect_boxes_for_image(self, pil_image: Image.Image, model):
        import numpy as np
        arr = np.array(pil_image)
        res = model.predict(source=arr, conf=YOLO_CONF_THRESHOLD, verbose=False, device="cpu")
        r0 = res[0]
        iw, ih = pil_image.size
        boxes_out: List[Box] = []

        names = getattr(r0, "names", None) or getattr(model, "names", None) or {}
        if isinstance(names, list):
            names = {i:n for i,n in enumerate(names)}

        if r0 and getattr(r0, "boxes", None) is not None:
            for b in r0.boxes:
                try:
                    cls_id = int(b.cls[0].item()) if hasattr(b,"cls") else None
                    xyxy = b.xyxy[0].tolist() if hasattr(b,"xyxy") else None
                    xyxy = self._sanitize_and_clip(xyxy, iw, ih)
                    if cls_id is None or xyxy is None: continue

                    nm_lower = str(names.get(cls_id,"")).lower() if names else ""
                    if cls_id not in (YOLO_UNLOCKED_ID, YOLO_LOCKED_ID):
                        if "unlock" in nm_lower: cls_id = YOLO_UNLOCKED_ID
                        elif "lock" in nm_lower: cls_id = YOLO_LOCKED_ID

                    x1,y1,x2,y2 = xyxy
                    boxes_out.append(Box(x1,y1,x2,y2,cls=cls_id,selected=False))
                except Exception as ex:
                    log_exc("_detect_boxes_iter", ex)
        return boxes_out, (names or {})

    def _set_classes_from_detections(self, cls_ids: List[int], names_map: Dict[int,str]):
        unique = sorted(set(cls_ids))
        self.classes = {}
        for cid in unique:
            name = names_map.get(cid, None)
            if name: name = str(name)
            if cid == YOLO_UNLOCKED_ID:
                name = "Unlocked" if not name else name
                color = PALETTE["canvasbg"]
                show = True
            elif cid == YOLO_LOCKED_ID:
                name = "Locked" if not name else name
                color = PALETTE["danger"]
                show = False
            else:
                name = name if name else f"class_{cid}"
                color = self._auto_color_for(cid)
                show = True
            self.classes[cid] = {"name": name, "color": color, "show": tk.BooleanVar(value=show)}
        if self.classes:
            self.var_new_cls.set(sorted(self.classes)[0])
        self._rebuild_visibility_ui()
        self._rebuild_newclass_ui()

    def on_prefill_once(self):
        if self.image is None:
            messagebox.showwarning("YOLO prefill", "Open an image first."); return
        if not (self.var_prefill.get() and _YOLO_OK and self.var_model.get().strip()):
            messagebox.showwarning("YOLO prefill", "Enable prefill and select a valid model file first."); return
        if self._prefill_running or self._scan_all_running:
            self._set_status("Another prefill is running…"); return

        self._prefill_running = True
        try:
            self._set_status("Prefilling with YOLO…"); self.update_idletasks()
            model = self._get_yolo_model(self.var_model.get().strip())
            new_boxes, names_map = self._detect_boxes_for_image(self.image, model)

            self._push_undo()
            self.boxes = new_boxes
            self._set_classes_from_detections([b.cls for b in new_boxes], names_map)

            self._yolo_prefilled = True
            self._remember_current()
            self.after_idle(self.redraw)
            self._set_status(f"Prefill complete. Boxes: {len(self.boxes)}")
        except BaseException as ex:
            log_exc("on_prefill_once", ex)
            try: messagebox.showerror("YOLO prefill failed", str(ex))
            except Exception: pass
        finally:
            self._prefill_running = False

    def on_scan_all(self):
        if not self.image_paths:
            messagebox.showwarning("Scan All", "Load images first."); return
        if not (self.var_prefill.get() and _YOLO_OK and self.var_model.get().strip()):
            messagebox.showwarning("Scan All", "Enable prefill and select a valid model file first."); return
        if self._prefill_running or self._scan_all_running:
            self._set_status("Another prefill is running…"); return

        self._scan_all_running = True
        processed = 0
        total_boxes = 0
        failed = 0
        names_map_for_current: Dict[int,str] = {}
        try:
            model = self._get_yolo_model(self.var_model.get().strip())
            curp = self.image_paths[self.image_idx] if (self.image_paths and 0 <= self.image_idx < len(self.image_paths)) else None
            for i, p in enumerate(self.image_paths):
                try:
                    fn = os.path.basename(p)
                    self._set_status(f"Scanning {i+1}/{len(self.image_paths)}: {fn}")
                    self.update_idletasks()
                    im = Image.open(p).convert("RGB")
                    boxes, names_map = self._detect_boxes_for_image(im, model)
                    snap = [(b.x1,b.y1,b.x2,b.y2,b.cls) for b in boxes]
                    self.annotations[p] = snap
                    self._write_yolo_txt_for_path(p, boxes)
                    total_boxes += len(boxes)
                    processed += 1
                    if curp and p == curp:
                        names_map_for_current = names_map
                except BaseException as ex:
                    failed += 1
                    log_exc("scan_all_item", ex)

            if self.image_paths and 0 <= self.image_idx < len(self.image_paths):
                curp = self.image_paths[self.image_idx]
                if curp in self.annotations:
                    self._push_undo()
                    self._restore_from_snapshot(self.annotations[curp])
                    self._yolo_prefilled = True
                    cls_ids = [t[4] for t in self.annotations[curp]]
                    self._set_classes_from_detections(cls_ids, names_map_for_current)
                    self._remember_current()
                    self.redraw()

            self._set_status(f"Scan All done. Files: {processed}, Boxes: {total_boxes}" + (f", Failed: {failed}" if failed else ""))
            try:
                messagebox.showinfo("Scan All", f"Completed.\nProcessed: {processed}\nBoxes total: {total_boxes}\nFailed: {failed}")
            except Exception:
                pass
        except BaseException as ex:
            log_exc("on_scan_all", ex)
            try: messagebox.showerror("Scan All failed", str(ex))
            except Exception: pass
        finally:
            self._scan_all_running = False

    # ---------- VIEW / ZOOM / PAN ----------
    def canvas_size(self) -> Tuple[int,int]:
        return (int(self.canvas.winfo_width()), int(self.canvas.winfo_height()))

    def _compute_base_scale(self):
        if self.image is None: return
        cw, ch = self.canvas_size()
        iw, ih = self.image.size
        self.base_scale = max(1e-9, min(cw / iw, ch / ih))
        self.scale = self.base_scale * self.zoom

    def _recenter_fit(self):
        if self.image is None: return
        self._compute_base_scale()
        cw, ch = self.canvas_size()
        iw, ih = self.image.size
        disp_w, disp_h = iw * self.scale, ih * self.scale
        self.offset_x = (cw - disp_w) / 2.0
        self.offset_y = (ch - disp_h) / 2.0

    def _clamp_offsets(self):
        if self.image is None: return
        cw, ch = self.canvas_size()
        iw, ih = self.image.size
        disp_w, disp_h = iw * self.scale, ih * self.scale
        def clamp(off, canvas, disp):
            if disp <= canvas:
                return (canvas - disp) / 2.0
            min_off = canvas - disp
            max_off = 0.0
            return min(max(off, min_off), max_off)
        self.offset_x = clamp(self.offset_x, cw, disp_w)
        self.offset_y = clamp(self.offset_y, ch, disp_h)

    def _clear_cache(self):
        self._cached_disp_size = None
        self._cached_pil = None
        self._cached_photo = None

    def _ensure_image_surface(self):
        if self.image is None: return
        iw, ih = self.image.size
        disp_w, disp_h = int(max(1, iw * self.scale)), int(max(1, ih * self.scale))
        need_resize = (self._cached_disp_size != (disp_w, disp_h)) or (self._cached_pil is None)

        if need_resize:
            self._cached_pil = self.image.resize((disp_w, disp_h), Image.NEAREST)
            self._cached_photo = ImageTk.PhotoImage(self._cached_pil)
            self._cached_disp_size = (disp_w, disp_h)

        if self._image_item is None:
            self._image_item = self.canvas.create_image(int(self.offset_x), int(self.offset_y),
                                                        image=self._cached_photo, anchor="nw", tags=("img",))
        else:
            self.canvas.itemconfig(self._image_item, image=self._cached_photo)
            self.canvas.coords(self._image_item, int(self.offset_x), int(self.offset_y))

    def zoom_step(self, factor: float, anchor: str="center", at_canvas_xy: Optional[Tuple[int,int]]=None):
        if self.image is None: return
        self._compute_base_scale()
        cw, ch = self.canvas_size()
        if anchor == "center" or at_canvas_xy is None:
            cx, cy = cw // 2, ch // 2
        else:
            cx, cy = at_canvas_xy
        ix, iy = self.canvas_to_img(cx, cy, clamp_inside=False)
        new_zoom = min(max(self.zoom * factor, self.min_zoom), self.max_zoom)
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        self.zoom = new_zoom
        self.scale = self.base_scale * self.zoom
        self.offset_x = cx - ix * self.scale
        self.offset_y = cy - iy * self.scale
        self._clamp_offsets()
        self._ensure_image_surface()
        self.redraw()

    def fit_to_screen(self):
        self.zoom = 1.0
        self._recenter_fit()
        self._ensure_image_surface()
        self.redraw()

    def on_ctrl_wheel(self, event):
        step = 1.15 if event.delta > 0 else 1/1.15
        self.zoom_step(step, at_canvas_xy=(event.x, event.y))

    def on_ctrl_wheel_linux(self, event, direction: int):
        step = 1.15 if direction > 0 else 1/1.15
        self.zoom_step(step, at_canvas_xy=(event.x, event.y))

    # --- Two-finger / wheel panning ---
    def on_pan_wheel(self, event):
        if self.image is None or self.control_held: return
        dy = (event.delta / 120.0) * PAN_PIXELS_PER_NOTCH
        self.offset_y += dy
        self._clamp_offsets()
        self._ensure_image_surface()
        self.redraw()

    def on_pan_wheel_h(self, event):
        if self.image is None: return
        dx = (event.delta / 120.0) * PAN_PIXELS_PER_NOTCH
        self.offset_x += dx
        self._clamp_offsets()
        self._ensure_image_surface()
        self.redraw()

    def on_pan_wheel_linux(self, event, direction: int):
        if self.image is None: return
        dy = direction * PAN_PIXELS_PER_NOTCH
        self.offset_y += dy
        self._clamp_offsets()
        self._ensure_image_surface()
        self.redraw()

    def on_pan_wheel_linux_h(self, event, direction: int):
        if self.image is None: return
        dx = direction * PAN_PIXELS_PER_NOTCH
        self.offset_x += dx
        self._clamp_offsets()
        self._ensure_image_surface()
        self.redraw()

    # ---------- drawing ----------
    def redraw(self):
        if self.image is None:
            self.canvas.delete("overlay")
            self._update_counts()
            return

        self._compute_base_scale()
        self._clamp_offsets()
        self._ensure_image_surface()
        self.canvas.delete("overlay")

        # Boxes
        for idx, box in enumerate(self.boxes):
            if box.cls not in self.classes:  # removed class
                continue
            if not self.classes[box.cls]["show"].get():
                continue
            x1, y1 = self.img_to_canvas(box.x1, box.y1)
            x2, y2 = self.img_to_canvas(box.x2, box.y2)
            outline = self.classes[box.cls]["color"]
            if box.selected: outline = PALETTE["warning"]
            self.canvas.create_rectangle(x1,y1,x2,y2, outline=outline, width=2,
                                         tags=(f"box-{idx}","box","overlay"))
            self.canvas.tag_bind(f"box-{idx}", "<Button-1>", lambda e, i=idx: self.select_box(i))

        # Handles for selected (only when exactly one is selected and visible)
        selected_count = sum(1 for b in self.boxes if b.selected)
        if selected_count == 1:
            sel_idx = self._selected_index()
            if sel_idx is not None and 0 <= sel_idx < len(self.boxes):
                b = self.boxes[sel_idx]
                if b.cls in self.classes and self.classes[b.cls]["show"].get():
                    self._draw_handles_for(sel_idx, b)

        # Overlays
        self._draw_crosshair()
        self._draw_cursor_plus()

        self._update_counts()

        try:
            if self.crosshair_on.get():
                self.cross_btn.configure(text="✚  Cross-hair ON", style="ToggleOn.TButton")
            else:
                self.cross_btn.configure(text="✚  Cross-hair OFF", style="TButton")
        except Exception:
            pass

    def _clear_marquee(self):
        if self.marquee_id is not None:
            try:
                self.canvas.delete(self.marquee_id)
            except Exception:
                pass
            self.marquee_id = None

    def _update_marquee(self, event):
        """Draw/refresh the marquee selection rectangle (canvas coords)."""
        if self.image is None:
            return
        self._clear_marquee()
        sx, sy = self.img_to_canvas(*self.marquee_start)
        cx, cy = event.x, event.y
        self.marquee_id = self.canvas.create_rectangle(
            sx, sy, cx, cy,
            outline=PALETTE["accent"], width=2, dash=(5, 3),
            tags=("overlay", "marquee"),
        )

    @staticmethod
    def _norm_rect(x1, y1, x2, y2):
        """Normalize rectangle so x1<=x2, y1<=y2."""
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        return x1, y1, x2, y2

    @staticmethod
    def _rects_intersect(a, b):
        """Axis-aligned intersection check. a/b are (x1,y1,x2,y2) inclusive-like."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return (ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1)

    def _draw_crosshair(self):
        self.canvas.delete("xhair")
        if not self.crosshair_on.get(): return
        x, y = self.mouse_canvas_xy
        cw, ch = self.canvas_size()
        self.canvas.create_line(x, 0, x, ch, fill=PALETTE["crosshair"], width=1, tags=("xhair","overlay"))
        self.canvas.create_line(0, y, cw, y, fill=PALETTE["crosshair"], width=1, tags=("xhair","overlay"))

    def _draw_cursor_plus(self):
        self.canvas.delete("cursorplus")
        if not self.cursor_hidden: return
        x, y = self.mouse_canvas_xy
        arm = 6
        self.canvas.create_line(x, y - arm, x, y + arm, fill="#000000", width=1, tags=("cursorplus","overlay"))
        self.canvas.create_line(x - arm, y, x + arm, y, fill="#000000", width=1, tags=("cursorplus","overlay"))

    def _selected_index(self) -> Optional[int]:
        for i, b in enumerate(self.boxes):
            if b.selected: return i
        return None

    def _draw_handles_for(self, idx: int, b: Box):
        x1c, y1c = self.img_to_canvas(b.x1, b.y1)
        x2c, y2c = self.img_to_canvas(b.x2, b.y2)
        mx, my = (x1c + x2c)//2, (y1c + y2c)//2
        centers = {
            "nw": (x1c, y1c), "n": (mx, y1c), "ne": (x2c, y1c),
            "w":  (x1c, my),               "e":  (x2c, my),
            "sw": (x1c, y2c), "s": (mx, y2c), "se": (x2c, y2c),
        }
        r = HANDLE_SIZE // 2
        for name, (cx, cy) in centers.items():
            self.canvas.create_rectangle(cx-r, cy-r, cx+r, cy+r,
                                         fill=PALETTE["warning"], outline=PALETTE["outline"],
                                         tags=(f"hdl-{name}", "handle", "overlay"))
            self.canvas.tag_bind(f"hdl-{name}", "<Button-1>",
                                 lambda e, i=idx, h=name: self._start_resize(i, h, e))

    # ---------- coords helpers ----------
    def img_to_canvas(self, x:int, y:int)->Tuple[int,int]:
        return int(self.offset_x + x*self.scale), int(self.offset_y + y*self.scale)

    def canvas_to_img(self, x:int, y:int, clamp_inside: bool=True)->Tuple[int,int]:
        if self.image is None: return 0,0
        iw, ih = self.image.size
        xi = (x - self.offset_x) / self.scale
        yi = (y - self.offset_y) / self.scale
        if clamp_inside:
            xi = max(0, min(iw-1, int(xi)))
            yi = max(0, min(ih-1, int(yi)))
        return int(xi), int(yi)

    def _update_counts(self):
        total = len(self.boxes)
        visible = sum(1 for b in self.boxes if b.cls in self.classes and self.classes[b.cls]["show"].get())
        fname = os.path.basename(self.image_paths[self.image_idx]) if (self.image_paths and self.image_idx>=0) else "—"
        self.lbl_counts.config(text=f"File: {fname}\nBoxes: {total} (Visible: {visible})")

    # ---------- interactions ----------
    def on_canvas_enter(self, _event=None):
        if self.image is not None:
            self.canvas.configure(cursor="none")
            self.cursor_hidden = True
            self._draw_cursor_plus()

    def on_canvas_leave(self, _event=None):
        self.canvas.configure(cursor="arrow")
        self.cursor_hidden = False
        self.canvas.delete("cursorplus")
        self.canvas.delete("xhair")

    def on_ctrl_down(self, _event=None):
        self.control_held = True

    def on_ctrl_up(self, _event=None):
        self.control_held = False
        if self.panning:
            self.panning = False

    def on_shift_down(self, _event=None): self.shift_held = True
    def on_shift_up(self, _event=None):   self.shift_held = False

    def on_alt_down(self, _event=None): self.alt_held = True
    def on_alt_up(self, _event=None):   self.alt_held = False

    def _on_number_hotkey(self, digit:int):
        if not self.classes:
            self._set_status("No labels yet.")
            return
        ids_sorted = sorted(self.classes)
        idx = digit - 1
        if 0 <= idx < len(ids_sorted):
            cls_id = ids_sorted[idx]
            self.var_new_cls.set(cls_id)
            nm = self.classes[cls_id]["name"]
            self._set_status(f'Class set to {nm} ({cls_id}) via hotkey {digit}')
        else:
            self._set_status(f"No class for hotkey {digit}.")

    def on_press(self, event):
        if self.image is None: return
        self.mouse_canvas_xy = (event.x, event.y)
        self._draw_cursor_plus()
        if self.crosshair_on.get(): self._draw_crosshair()

        # Ctrl+Drag -> pan
        if self.control_held:
            self.panning = True
            self.pan_start_canvas = (event.x, event.y)
            self.pan_start_offset = (self.offset_x, self.offset_y)
            return

        handle_hit = self._handle_hit_at_canvas(event.x, event.y)
        if handle_hit is not None:
            idx, hname = handle_hit
            self._start_resize(idx, hname, event)
            return

        imgx, imgy = self.canvas_to_img(event.x, event.y)
        hit = self._hit_test_visible(imgx, imgy)
        # 
        if not self.shift_held and hit is None:
            if any(b.selected for b in self.boxes):
                for b in self.boxes:
                    b.selected = False
                self.redraw() 
        # --- SHIFT logic: additive selection or marquee selection
        if self.shift_held:
            if hit is not None:
                # Additive select (don't toggle off)
                if not self.boxes[hit].selected:
                    self.boxes[hit].selected = True
                    self._set_status("Added to selection.")
                    self.redraw()
                else:
                    # Already selected: start group move immediately
                    self._push_undo()
                    self.moving = True
                    self.move_start_img = (imgx, imgy)
                    self.move_selected_indices = [i for i, b in enumerate(self.boxes) if b.selected]
                    self.move_start_boxes = {i: (self.boxes[i].x1, self.boxes[i].y1,
                                                 self.boxes[i].x2, self.boxes[i].y2)
                                             for i in self.move_selected_indices}
                return
            else:
                # Start marquee selection
                self.marquee_selecting = True
                self.marquee_start = (imgx, imgy)
                self._update_marquee(event)
                return

        # --- no SHIFT
        if hit is not None:
            # Start move: select clicked if it wasn't selected; else move all selected
            self._push_undo()
            if not self.boxes[hit].selected:
                for b in self.boxes: b.selected = False
                self.boxes[hit].selected = True

            self.moving = True
            self.move_start_img = (imgx, imgy)
            self.move_selected_indices = [i for i, b in enumerate(self.boxes) if b.selected]
            self.move_start_boxes = {i: (self.boxes[i].x1, self.boxes[i].y1,
                                         self.boxes[i].x2, self.boxes[i].y2)
                                     for i in self.move_selected_indices}
            return

        # Start a new annotation box
        self.dragging = True
        self.drag_start = (imgx, imgy)
        self._update_rubber(event)

    def on_drag(self, event):
        if self.image is None: return

        self.mouse_canvas_xy = (event.x, event.y)
        self._draw_cursor_plus()
        if self.crosshair_on.get(): self._draw_crosshair()

        # Pan while Ctrl held
        if self.panning:
            sx, sy = self.pan_start_canvas
            dx = event.x - sx
            dy = event.y - sy
            self.offset_x = self.pan_start_offset[0] + dx
            self.offset_y = self.pan_start_offset[1] + dy
            self._clamp_offsets()
            self._ensure_image_surface()
            self.redraw()
            return

        if self.resizing and self.resize_idx is not None:
            imgx, imgy = self.canvas_to_img(event.x, event.y)
            self._apply_resize(self.resize_idx, self.resize_handle, imgx, imgy)
            self._remember_current()
            self.redraw()
            return

        # --- marquee selection drag
        if self.marquee_selecting:
            self._update_marquee(event)
            return

        # Group moving
        if self.moving and self.move_selected_indices:
            sx, sy = self.move_start_img
            imgx, imgy = self.canvas_to_img(event.x, event.y)
            dx, dy = imgx - sx, imgy - sy
            for i in self.move_selected_indices:
                b = self.boxes[i]
                x1, y1, x2, y2 = self.move_start_boxes[i]
                b.x1, b.y1, b.x2, b.y2 = x1, y1, x2, y2
                b.move_by(dx, dy, self.image.size)
            self._remember_current()
            self.redraw()
            return

        # Annotation rubber band
        if self.dragging:
            self._update_rubber(event)

    def on_release(self, event):
        if self.image is None: return
        self.mouse_canvas_xy = (event.x, event.y)
        self._draw_cursor_plus()
        if self.crosshair_on.get(): self._draw_crosshair()

        if self.panning:
            self.panning = False
            return

        if self.resizing:
            self.resizing = False
            self.resize_idx = None
            self.resize_handle = None
            self.resize_start_box = None
            self._remember_current()
            self.redraw()
            return

        # --- finish marquee selection
        if self.marquee_selecting:
            self.marquee_selecting = False
            sx, sy = self.marquee_start
            ex, ey = self.canvas_to_img(event.x, event.y)
            x1, y1, x2, y2 = self._norm_rect(sx, sy, ex, ey)
            selected_now = 0
            for b in self.boxes:
                # Only consider visible boxes for hit-testing
                if b.cls in self.classes and not self.classes[b.cls]["show"].get():
                    continue
                if self._rects_intersect((b.x1, b.y1, b.x2, b.y2), (x1, y1, x2, y2)):
                    if not b.selected:
                        b.selected = True
                        selected_now += 1
            self._clear_marquee()
            self.redraw()
            self._set_status(f"Selected {selected_now} box(es).")
            return

        if self.moving:
            self.moving = False
            self.move_selected_indices = []
            self.move_start_boxes = {}
            self.move_start_img = (0, 0)
            self._remember_current()
            self.redraw()
            return

        if self.dragging:
            self.dragging = False
            sx, sy = self.drag_start
            ex, ey = self.canvas_to_img(event.x, event.y)
            if self.alt_held:
                dx, dy = ex - sx, ey - sy
                side = min(abs(dx), abs(dy))
                ex = sx + side * (1 if dx >= 0 else -1)
                ey = sy + side * (1 if dy >= 0 else -1)

            if not self.classes:
                if self.rubber_id is not None:
                    self.canvas.delete(self.rubber_id); self.rubber_id = None
                messagebox.showwarning("No labels", "Add a label first (CLASSES → Add label…) or run YOLO Prefill.")
                self._set_status("Cannot create box: no labels.")
                return

            cls_selected = self.var_new_cls.get()
            if cls_selected not in self.classes:
                if self.rubber_id is not None:
                    self.canvas.delete(self.rubber_id); self.rubber_id = None
                messagebox.showwarning("Label missing", "Selected label no longer exists.")
                return

            nb = Box(sx, sy, ex, ey, cls=cls_selected, selected=False)
            if nb.size_ok():
                self._push_undo()
                self.boxes.append(nb)
                self._remember_current()
            if self.rubber_id is not None:
                self.canvas.delete(self.rubber_id); self.rubber_id = None
            self.redraw()

    def on_mouse_move(self, event):
        self.mouse_canvas_xy = (event.x, event.y)
        self._draw_cursor_plus()
        if self.crosshair_on.get():
            self._draw_crosshair()

    def _start_resize(self, idx:int, handle:str, _event):
        self._push_undo()  # snapshot before resize
        self.select_box(idx)
        self.resizing = True
        self.resize_idx = idx
        self.resize_handle = handle
        b = self.boxes[idx]
        self.resize_start_box = (b.x1, b.y1, b.x2, b.y2)

    def _update_rubber(self, event):
        if self.image is None: return
        if self.rubber_id is not None:
            self.canvas.delete(self.rubber_id); self.rubber_id = None
        sx, sy = self.img_to_canvas(*self.drag_start)
        cx, cy = event.x, event.y
        if self.alt_held:
            dx, dy = cx - sx, cy - sy
            side = min(abs(dx), abs(dy))
            cx = sx + side * (1 if dx >= 0 else -1)
            cy = sy + side * (1 if dy >= 0 else -1)
        self.rubber_id = self.canvas.create_rectangle(sx, sy, cx, cy, outline=PALETTE["warning"], width=2, dash=(3,2), tags=("overlay",))
        self._draw_crosshair()
        self._draw_cursor_plus()

    def _hit_test_visible(self, imgx:int, imgy:int)->Optional[int]:
        for i in range(len(self.boxes)-1, -1, -1):
            b = self.boxes[i]
            if b.cls in self.classes and not self.classes[b.cls]["show"].get():
                continue
            if b.contains(imgx, imgy):
                return i
        return None

    def _handle_hit_at_canvas(self, cx:int, cy:int) -> Optional[Tuple[int,str]]:
        # Only allow handle hit-testing when exactly one box is selected
        if sum(1 for b in self.boxes if b.selected) != 1:
            return None

        sel = self._selected_index()
        if sel is None: return None
        b = self.boxes[sel]
        if b.cls not in self.classes or not self.classes[b.cls]["show"].get():
            return None
        x1c, y1c = self.img_to_canvas(b.x1, b.y1)
        x2c, y2c = self.img_to_canvas(b.x2, b.y2)
        mx, my = (x1c + x2c)//2, (y1c + y2c)//2
        centers = {
            "nw": (x1c, y1c), "n": (mx, y1c), "ne": (x2c, y1c),
            "w":  (x1c, my),               "e":  (x2c, my),
            "sw": (x1c, y2c), "s": (mx, y2c), "se": (x2c, y2c),
        }
        r = HANDLE_SIZE // 2
        for name, (hx, hy) in centers.items():
            if hx - r <= cx <= hx + r and hy - r <= cy <= hy + r:
                return (sel, name)
        return None

    def _apply_resize(self, idx:int, handle:str, imgx:int, imgy:int):
        if self.image is None or self.resize_start_box is None: return
        b = self.boxes[idx]
        iw, ih = self.image.size
        sx1, sy1, sx2, sy2 = self.resize_start_box
        x1, y1, x2, y2 = sx1, sy1, sx2, sy2

        tx = max(0, min(iw-1, imgx))
        ty = max(0, min(ih-1, imgy))

        def clamp_min_w_left(nx1):  return min(max(nx1, 0), x2 - MIN_SIDE)
        def clamp_min_w_right(nx2): return max(min(nx2, iw-1), x1 + MIN_SIDE)
        def clamp_min_h_top(ny1):   return min(max(ny1, 0), y2 - MIN_SIDE)
        def clamp_min_h_bot(ny2):   return max(min(ny2, ih-1), y1 + MIN_SIDE)

        if handle in ("n","s","w","e") and not self.alt_held:
            if handle == "n": y1 = clamp_min_h_top(ty)
            elif handle == "s": y2 = clamp_min_h_bot(ty)
            elif handle == "w": x1 = clamp_min_w_left(tx)
            elif handle == "e": x2 = clamp_min_w_right(tx)
        else:
            if handle in ("nw","ne","sw","se") and self.alt_held:
                if handle == "nw": ax, ay = sx2, sy2; sx = -1; sy = -1
                elif handle == "ne": ax, ay = sx1, sy2; sx = +1; sy = -1
                elif handle == "sw": ax, ay = sx2, sy1; sx = -1; sy = +1
                else: ax, ay = sx1, sy1; sx = +1; sy = +1
                s = min(abs(tx - ax), abs(ty - ay))
                s_max_x = ax if sx < 0 else (iw-1 - ax)
                s_max_y = ay if sy < 0 else (ih-1 - ay)
                s = max(MIN_SIDE, min(s, s_max_x, s_max_y))
                nx = ax + sx * s; ny = ay + sy * s
                if handle == "nw": x1, y1, x2, y2 = nx, ny, ax, ay
                elif handle == "ne": x1, y1, x2, y2 = ax, ny, nx, ay
                elif handle == "sw": x1, y1, x2, y2 = nx, ay, ax, ny
                else:                x1, y1, x2, y2 = ax, ay, nx, ny
            else:
                if handle == "nw":
                    x1 = clamp_min_w_left(tx); y1 = clamp_min_h_top(ty)
                elif handle == "ne":
                    x2 = clamp_min_w_right(tx); y1 = clamp_min_h_top(ty)
                elif handle == "sw":
                    x1 = clamp_min_w_left(tx); y2 = clamp_min_h_bot(ty)
                elif handle == "se":
                    x2 = clamp_min_w_right(tx); y2 = clamp_min_h_bot(ty)
                else:
                    if handle == "n": y1 = clamp_min_h_top(ty)
                    elif handle == "s": y2 = clamp_min_h_bot(ty)
                    elif handle == "w": x1 = clamp_min_w_left(tx)
                    elif handle == "e": x2 = clamp_min_w_right(tx)

        b.x1, b.y1, b.x2, b.y2 = int(x1), int(y1), int(x2), int(y2)

    # ---------- utility ----------
    def select_box(self, idx:int):
        # single selection helper (used on direct clicks)
        for i,b in enumerate(self.boxes):
            b.selected = (i==idx)
        self.redraw()

    def on_right_click(self, event):
        if self.image is None or self.ctx is None: return
        imgx, imgy = self.canvas_to_img(event.x, event.y)
        hit = self._hit_test_visible(imgx, imgy)
        if hit is not None:
            for i,b in enumerate(self.boxes):
                b.selected = (i==hit)
            try:
                self.ctx.tk_popup(event.x_root, event.y_root)
            finally:
                self.ctx.grab_release()

    def on_delete_selected(self, event=None):
        sel_count = sum(1 for b in self.boxes if b.selected)
        if sel_count == 0:
            self._set_status("Delete: no selection.")
            return
        self._push_undo()
        self.boxes = [b for b in self.boxes if not b.selected]
        self._remember_current()
        self._set_status(f"Deleted {sel_count} selected box(es).")
        self.redraw()

    def clear_boxes(self):
        if not self.boxes:
            self._set_status("No boxes to clear.")
            return

        cnt = len(self.boxes)
        msg = (f"You're about to delete all {cnt} box(es) on this image.\n"
               f"This action can be undone with Ctrl+Z.")
        choice = self._modal_choice(
            "Clear all boxes?",
            msg,
            buttons=[
                ("Delete boxes", "Danger.TButton", "delete"),  # primary (Enter)
                ("Keep boxes",   "TButton",         "keep"),
                ("Cancel",       "TButton",          None),
            ],
            width=520,
        )

        if choice != "delete":
            self._set_status("Clear cancelled.")
            return

        self._push_undo()
        self.boxes.clear()
        self._remember_current()
        self._set_status("Cleared boxes.")
        self.redraw()




    def set_selected_class(self, cls_id:int):
        if cls_id not in self.classes:
            messagebox.showwarning("Unknown class", f"Class id {cls_id} does not exist.")
            return
        changed = False
        for b in self.boxes:
            if b.selected and b.cls != cls_id:
                if not changed: self._push_undo()
                b.cls = cls_id; changed = True
        if changed:
            self._remember_current()
            self._set_status(f"Changed selected box to class {cls_id} ({self.classes[cls_id]['name']}).")
            self.redraw()

    def nudge_selected(self, dx:int, dy:int):
        if self.image is None: return
        sel_idxs = [i for i, b in enumerate(self.boxes) if b.selected]
        if not sel_idxs:
            return
        self._push_undo()
        for i in sel_idxs:
            self.boxes[i].move_by(dx, dy, self.image.size)
        self._remember_current()
        self.redraw()

    # ---- copy / paste (group-aware) ----
    def copy_selected(self):
        sel_idxs = [i for i, b in enumerate(self.boxes) if b.selected]
        if not sel_idxs:
            self._set_status("Copy: no selection.")
            return
        self.copied_box = [(self.boxes[i].x1, self.boxes[i].y1, self.boxes[i].x2, self.boxes[i].y2, self.boxes[i].cls)
                           for i in sel_idxs]
        self.paste_count = 0  # reset so the NEXT paste gets an offset
        self._set_status(f"Copied {len(sel_idxs)} box(es).")

    def paste_copied(self):
        if self.image is None or self.copied_box is None:
            self._set_status("Paste: nothing to paste.")
            return
        if not self.classes:
            self._set_status("Paste failed: no labels exist.")
            return

        # Normalize to list
        boxes_to_paste = self.copied_box
        if isinstance(boxes_to_paste, tuple):
            boxes_to_paste = [boxes_to_paste]

        iw, ih = self.image.size
        self.paste_count += 1
        off = self.paste_nudge * (self.paste_count % 6)

        for b in self.boxes:
            b.selected = False

        self._push_undo()
        pasted_any = False
        for (x1, y1, x2, y2, cls) in boxes_to_paste:
            if cls not in self.classes:
                cls = sorted(self.classes)[0]
            nx1 = max(0, min(iw - 1, x1 + off))
            ny1 = max(0, min(ih - 1, y1 + off))
            w = x2 - x1
            h = y2 - y1
            nx2 = max(0, min(iw - 1, nx1 + w))
            ny2 = max(0, min(ih - 1, ny1 + h))
            if nx2 - nx1 >= MIN_SIDE and ny2 - ny1 >= MIN_SIDE:
                nb = Box(nx1, ny1, nx2, ny2, cls=cls, selected=True)
                self.boxes.append(nb)
                pasted_any = True

        if pasted_any:
            self._remember_current()
            self.redraw()
            self._set_status(f"Pasted {len(boxes_to_paste)} box(es) (selected).")
        else:
            self._set_status("Paste failed: out of bounds.")

    # ---------- manual save ----------
    def _do_save(self, silent: bool = False):
        if self.image is None or not self.image_paths:
            if not silent:
                messagebox.showerror("Save", "No image loaded.")
            return False
    
        src_path = self.image_paths[self.image_idx]
    
        try:
            # Write YOLO labels only (no image duplication)
            self._write_yolo_txt_for_path(src_path, self.boxes)
        except Exception as ex:
            if not silent:
                messagebox.showerror("Save label failed", str(ex))
            return False
    
        self.dirty = False
        if not silent:
            out_txt = self._yolo_txt_path_for(src_path)
            self._set_status(f"Saved labels -> {out_txt}")
            try:
                messagebox.showinfo("Saved", f"Labels:\n  {out_txt}")
            except Exception:
                pass
        return True
    
    def on_save(self):
        self._do_save(silent=False)

# ---------- main ----------
if __name__ == "__main__":
    app = LabelerApp()
    app.mainloop()
