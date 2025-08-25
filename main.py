import os
import glob
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import dearpygui.dearpygui as dpg


def palette_grayscale() -> np.ndarray:
    g = np.arange(256, dtype=np.uint8)
    return np.stack([g, g, g], axis=1)

def palette_fire() -> np.ndarray:
    x = np.linspace(0, 1, 256)
    r = np.clip(255 * (1.5 * x), 0, 255)
    g = np.clip(255 * np.maximum(0, (x - 0.3) / 0.7), 0, 255)
    b = np.clip(255 * np.maximum(0, (x - 0.8) / 0.2), 0, 255)
    return np.stack([r, g, b], axis=1).astype(np.uint8)

PALETTES = {
    "Grayscale": palette_grayscale(),
    "Fire": palette_fire(),
}
ENTROPY_PALETTE = palette_fire()

def shannon_entropy_block(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    counts = np.bincount(block, minlength=256).astype(np.float64)
    p = counts / block.size
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))

def entropy_per_block(data: np.ndarray, block_size: int) -> np.ndarray:
    n = data.size
    block_size = max(1, int(block_size))
    num_blocks = (n + block_size - 1) // block_size
    out = np.empty(num_blocks, dtype=np.float32)
    for i in range(num_blocks):
        s = i * block_size
        e = min(s + block_size, n)
        out[i] = shannon_entropy_block(data[s:e])
    return out

MAX_PIXELS = 8_000_000  # ~8MP cap for Byte Map to avoid VRAM spikes

def render_byte_map(data: np.ndarray, cols: int, palette_name: str) -> Tuple[np.ndarray, int, int]:
    palette = PALETTES.get(palette_name, PALETTES["Grayscale"])
    n = int(data.size)
    cols = max(1, int(cols))
    cols = min(cols, n) if n > 0 else cols

    rows = (n + cols - 1) // cols if cols else 0
    # Adjust for large files by subsampling
    step = 1
    if rows * cols > MAX_PIXELS and n > 0:
        step = int(np.ceil((rows * cols) / MAX_PIXELS))
        data = data[::step]
        n = int(data.size)
        rows = (n + cols - 1) // cols
        cols = min(cols, int(np.sqrt(n) + 0.5))  # Maintain aspect ratio

    rgb = palette[data]
    pad = rows * cols - n
    if pad > 0:
        rgb = np.vstack([rgb, np.zeros((pad, 3), dtype=np.uint8)])
    img = rgb.reshape(rows, cols, 3)
    return (img.astype(np.float32) / 255.0, cols, rows)

def render_entropy_map(data: np.ndarray, cols_blocks: int, block_size: int) -> Tuple[np.ndarray, int, int, np.ndarray]:
    ent = entropy_per_block(data, max(1, int(block_size)))
    num_blocks = int(ent.size)
    cols = max(1, int(cols_blocks))
    rows = (num_blocks + cols - 1) // cols

    ent_idx = (np.clip(ent, 0, 8) * (255.0 / 8.0)).astype(np.uint8)
    pad = rows * cols - num_blocks
    if pad > 0:
        ent_idx = np.concatenate([ent_idx, np.zeros(pad, dtype=np.uint8)])

    rgb = ENTROPY_PALETTE[ent_idx]
    img = rgb.reshape(rows, cols, 3)
    ent_img_vals = ent_idx.reshape(rows, cols) * (8.0 / 255.0)

    return (img.astype(np.float32) / 255.0, cols, rows, ent_img_vals)

@dataclass
class AppState:
    path: Optional[str] = None
    data: Optional[np.ndarray] = None  # uint8
    mode: str = "Entropy Map"
    cols: int = 128
    block_size: int = 4096
    palette: str = "Grayscale"
    zoom: float = 8.0
    tex_id: Optional[int] = None
    tex_w: int = 0
    tex_h: int = 0
    ent_values: Optional[np.ndarray] = None
    busy: bool = False

STATE = AppState()

def set_status(msg: str):
    dpg.set_value("status_text", msg.replace('\n', ' | '))

def update_texture(img_float: np.ndarray, w: int, h: int):
    """Create/update GPU texture using raw float32 RGB buffer (contiguous)."""
    try:
        buf = np.ascontiguousarray(img_float.astype(np.float32))
        if buf.ndim != 3 or buf.shape[2] != 3:
            raise ValueError(f"Texture expects HxWx3 float32, got {buf.shape}")
        flat = buf.ravel()
        if STATE.tex_id is None or not dpg.does_item_exist(STATE.tex_id):
            with dpg.texture_registry(show=False):
                STATE.tex_id = dpg.add_raw_texture(w, h, flat, format=dpg.mvFormat_Float_rgb)
        else:
            if w != STATE.tex_w or h != STATE.tex_h:
                try:
                    if dpg.does_item_exist(STATE.tex_id):
                        dpg.delete_item(STATE.tex_id)
                    with dpg.texture_registry(show=False):
                        STATE.tex_id = dpg.add_raw_texture(w, h, flat, format=dpg.mvFormat_Float_rgb)
                except Exception as e:
                    set_status(f"Texture recreation failed: {str(e)}")
                    return
            else:
                dpg.set_value(STATE.tex_id, flat)
        STATE.tex_w, STATE.tex_h = w, h
        dpg.configure_item("image_widget", texture_tag=STATE.tex_id, width=int(max(1, w) * STATE.zoom), height=int(max(1, h) * STATE.zoom))
    except Exception:
        set_status(f"Texture update failed: {traceback.format_exc().replace('\n', ' | ')}")

def compute_and_render():
    if STATE.data is None:
        return
    STATE.busy = True
    try:
        set_status("Rendering…")
        # Adjust cols dynamically based on file size for both modes
        if STATE.mode == "Byte Map":
            STATE.cols = max(1, min(1024, int(np.sqrt(STATE.data.size) + 0.5)))
        else:
            # For Entropy Map, use fewer columns to reflect block-based rendering
            num_blocks = (STATE.data.size + STATE.block_size - 1) // STATE.block_size
            STATE.cols = max(1, min(512, int(np.sqrt(num_blocks) + 0.5)))
        dpg.set_value("cols_input", STATE.cols)

        if STATE.mode == "Byte Map":
            img, w, h = render_byte_map(STATE.data, STATE.cols, STATE.palette)
            STATE.ent_values = None  # Clear entropy values
            update_texture(img, w, h)
            step = int(np.ceil((w * h) / MAX_PIXELS)) if w * h > MAX_PIXELS else 1
            set_status(f"Byte Map: {STATE.path} ({STATE.data.size:,} bytes) → {w}×{h} px (zoom {STATE.zoom:.2f}×{', subsampled' if step > 1 else ''})")
        else:
            img, w, h, ent_img = render_entropy_map(STATE.data, STATE.cols, STATE.block_size)
            STATE.ent_values = ent_img
            update_texture(img, w, h)
            nb = (STATE.data.size + STATE.block_size - 1) // STATE.block_size
            set_status(f"Entropy Map: {STATE.path} ({STATE.data.size:,} bytes) | blocks={nb:,}, block={STATE.block_size:,} B → {w}×{h} px (zoom {STATE.zoom:.2f}×)")
    except MemoryError:
        set_status("Out of memory during render. Try Entropy Map or fewer Columns.")
    except Exception:
        set_status(f"Render failed: {traceback.format_exc().replace('\n', ' | ')}")
    finally:
        STATE.busy = False

def open_file_dialog():
    if STATE.busy:
        return
    dpg.show_item("file_dialog")

def on_file_selected(sender, app_data):
    path = app_data.get("file_path_name")
    if not path and isinstance(app_data.get("selections"), dict) and app_data["selections"]:
        path = next(iter(app_data["selections"].values()))

    if not path:
        set_status("No file selected.")
        return

    candidates = []
    if any(ch in path for ch in "*?["):
        candidates = glob.glob(path)
        if not candidates:
            set_status(f"No files match: {path}")
            return
    else:
        candidates = [path]

    chosen = None
    for p in candidates:
        if os.path.isfile(p):
            chosen = p
            break
    if chosen is None:
        set_status(f"Not a file: {path}")
        return

    try:
        set_status(f"Loading file: {chosen}…")
        with open(chosen, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        STATE.path = chosen
        STATE.data = raw
        # Reset texture to avoid stale data
        if STATE.tex_id is not None and dpg.does_item_exist(STATE.tex_id):
            dpg.delete_item(STATE.tex_id)
            STATE.tex_id = None
        STATE.tex_w, STATE.tex_h = 0, 0
        # Set reasonable default columns based on file size
        STATE.cols = max(1, min(1024, int(np.sqrt(raw.size) + 0.5)))
        dpg.set_value("cols_input", STATE.cols)
        compute_and_render()
    except Exception:
        set_status(f"Error loading file: {traceback.format_exc().replace('\n', ' | ')}")

def set_zoom(z: float):
    STATE.zoom = float(max(0.1, min(z, 100.0)))
    dpg.configure_item("image_widget", width=int(max(1, STATE.tex_w) * STATE.zoom), height=int(max(1, STATE.tex_h) * STATE.zoom))

def fit_to_window():
    try:
        cw, ch = dpg.get_item_rect_size("image_container")
        cw = max(1, int(cw - 8))
        ch = max(1, int(ch - 8))
        if STATE.tex_w == 0 or STATE.tex_h == 0:
            return
        zx = cw / STATE.tex_w
        zy = ch / STATE.tex_h
        set_zoom(max(0.1, min(zx, zy)))
    except Exception:
        pass

def zoom_at_cursor(factor: float):
    try:
        cont_min = dpg.get_item_rect_min("image_container")
        mx, my = dpg.get_mouse_pos(local=False)
        local_x = mx - cont_min[0]
        local_y = my - cont_min[1]
        sx = dpg.get_x_scroll("image_container")
        sy = dpg.get_y_scroll("image_container")
        cx = sx + local_x
        cy = sy + local_y
    except Exception:
        cx = cy = None

    old = STATE.zoom
    set_zoom(old * factor)

    if cx is not None:
        try:
            scale = STATE.zoom / max(1e-6, old)
            new_sx = cx * scale - local_x
            new_sy = cy * scale - local_y
            # Clamp scroll values
            max_sx = max(0, STATE.tex_w * STATE.zoom - dpg.get_item_rect_size("image_container")[0])
            max_sy = max(0, STATE.tex_h * STATE.zoom - dpg.get_item_rect_size("image_container")[1])
            dpg.set_x_scroll("image_container", max(0.0, min(new_sx, max_sx)))
            dpg.set_y_scroll("image_container", max(0.0, min(new_sy, max_sy)))
        except Exception:
            pass

def on_mouse_wheel(sender, app_data):
    if not (dpg.is_item_hovered("image_widget") or dpg.is_item_hovered("image_container")):
        return
    # Normalize wheel delta (typically ±120 per scroll step)
    delta = float(app_data) / 120.0
    zoom_at_cursor(1.1 ** delta)

def on_double_click(sender, app_data):
    if dpg.is_item_hovered("image_widget") or dpg.is_item_hovered("image_container"):
        fit_to_window()

def on_mouse_move():
    if STATE.tex_id is None or STATE.data is None:
        dpg.set_value("hover_label", "")
        return
    if not dpg.is_item_hovered("image_widget"):
        dpg.set_value("hover_label", "")
        return
    try:
        mx, my = dpg.get_mouse_pos(local=False)
        ix, iy = dpg.get_item_rect_min("image_widget")
        x = mx - ix
        y = my - iy
        if x < 0 or y < 0:
            dpg.set_value("hover_label", "")
            return
        px = int(x / max(STATE.zoom, 1e-6))
        py = int(y / max(STATE.zoom, 1e-6))
        if px < 0 or py < 0 or px >= STATE.tex_w or py >= STATE.tex_h:
            dpg.set_value("hover_label", "")
            return

        if STATE.mode == "Byte Map":
            cols = STATE.tex_w
            idx = py * cols + px
            if idx < STATE.data.size:
                b = int(STATE.data[idx])
                txt = f"Offset: 0x{idx:08X} ({idx})\nByte: {b} (0x{b:02X})"
            else:
                txt = "(padding)"
        else:
            cols = STATE.tex_w
            bidx = py * cols + px
            bstart = bidx * STATE.block_size
            bend = min(bstart + STATE.block_size, STATE.data.size)
            ent = float(STATE.ent_values[py, px]) if STATE.ent_values is not None and py < STATE.ent_values.shape[0] and px < STATE.ent_values.shape[1] else float('nan')
            txt = f"Block: {bidx}\nRange: 0x{bstart:08X}–0x{bend:08X} ({bend-bstart} B)\nEntropy: {ent:.3f} bits/byte"
        dpg.set_value("hover_label", txt)
    except Exception:
        dpg.set_value("hover_label", "")

def zoom_in():
    set_zoom(STATE.zoom * 1.25)

def zoom_out():
    set_zoom(STATE.zoom / 1.25)

def zoom_1_1():
    set_zoom(1.0)

def on_controls_change(sender=None, app_data=None):
    if STATE.busy:
        return
    old_mode = STATE.mode
    STATE.mode = dpg.get_value("mode_combo")
    STATE.cols = max(1, min(int(dpg.get_value("cols_input")), 10000))  # Cap columns
    STATE.block_size = max(1, min(int(dpg.get_value("block_input")), 1000000))  # Cap block size
    STATE.palette = dpg.get_value("palette_combo")
    # Force texture cleanup when switching modes
    if old_mode != STATE.mode and STATE.tex_id is not None and dpg.does_item_exist(STATE.tex_id):
        dpg.delete_item(STATE.tex_id)
        STATE.tex_id = None
        STATE.tex_w, STATE.tex_h = 0, 0
    compute_and_render()

def build_ui():
    dpg.create_context()

    with dpg.window(label="BinVis-DPG", tag="main", width=1280, height=860):
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Open…", callback=open_file_dialog)
                dpg.add_menu_item(label="Quit", callback=lambda: dpg.stop_dearpygui())
            with dpg.menu(label="View"):
                dpg.add_menu_item(label="Compute / Refresh", callback=compute_and_render)
                dpg.add_menu_item(label="Fit to Window", callback=fit_to_window)
                dpg.add_menu_item(label="1:1", callback=zoom_1_1)

        with dpg.group(horizontal=True):
            with dpg.child_window(width=340, height=-1, border=True):
                dpg.add_text("Controls", bullet=True)
                dpg.add_combo(["Entropy Map", "Byte Map"], default_value=STATE.mode, tag="mode_combo", callback=on_controls_change)
                dpg.add_separator()
                dpg.add_input_int(label="Columns", default_value=STATE.cols, tag="cols_input", min_value=1, min_clamped=True, max_value=10000, max_clamped=True, callback=on_controls_change)
                dpg.add_input_int(label="Block size (B)", default_value=STATE.block_size, tag="block_input", min_value=1, min_clamped=True, max_value=1000000, max_clamped=True, callback=on_controls_change)
                dpg.add_combo(list(PALETTES.keys()), default_value=STATE.palette, tag="palette_combo", callback=on_controls_change)
                dpg.add_separator()
                dpg.add_text("Zoom:")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="+", width=32, callback=zoom_in)
                    dpg.add_button(label="−", width=32, callback=zoom_out)
                    dpg.add_button(label="Fit", width=50, callback=fit_to_window)
                    dpg.add_button(label="1:1", width=50, callback=zoom_1_1)
                dpg.add_separator()
                dpg.add_text("Status:")
                dpg.add_text("", tag="status_text")
                dpg.add_spacer(height=6)
                dpg.add_text("Hover:")
                dpg.add_input_text(multiline=True, readonly=True, width=300, height=160, tag="hover_label")

            with dpg.child_window(width=-1, height=-1, border=True, tag="image_container"):
                # Initialize placeholder with gray for visibility
                with dpg.texture_registry(show=False):
                    placeholder = np.full((1, 1, 3), 0.5, dtype=np.float32).ravel()
                    STATE.tex_id = dpg.add_raw_texture(1, 1, placeholder, format=dpg.mvFormat_Float_rgb)
                dpg.add_image(tag="image_widget", texture_tag=STATE.tex_id, width=8, height=8)

    with dpg.file_dialog(directory_selector=False, show=False, callback=on_file_selected, tag="file_dialog"):
        dpg.add_file_extension(".*", color=(200, 200, 200, 255))

    # Global handlers
    with dpg.handler_registry():
        dpg.add_mouse_wheel_handler(callback=on_mouse_wheel)
        dpg.add_mouse_double_click_handler(callback=on_double_click)
        dpg.add_mouse_move_handler(callback=on_mouse_move)

    dpg.set_primary_window("main", True)
    dpg.create_viewport(title="BinVis-DPG", width=1300, height=880)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    set_status("Open a file to visualize.")
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    build_ui()
