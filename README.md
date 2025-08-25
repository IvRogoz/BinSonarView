# BinSonarView
a fast, lightweight **binary file visualizer** built with [Dear PyGui]. 

It provides two complementary views:

- **Byte Map** — maps each byte value (0–255) to a color.
- **Entropy Map** — computes Shannon entropy per block and colors each block (0–8 bits/byte).

Both views support smooth zooming, 1:1, fit-to-window, palette switching, and hover tooltips (offset/byte for Byte Map; block range/entropy for Entropy Map). Large files are handled gracefully via an ~8MP texture cap and adaptive subsampling.

---

## Features

- **Two modes**: Byte Map & Entropy Map  
- **Smooth zoom** (mouse wheel), **1:1**, and **Fit to window**  
- **Hover tooltips**  
  - Byte Map: file offset + byte value (dec/hex)  
  - Entropy Map: block index, byte range, entropy (bits/byte)  
- **Palettes**: Grayscale, Fire (easy to extend)  
- **Large-file safety**: ~8MP texture cap with automatic subsampling  
- **Simple UI**: live controls for columns, block size, palette, and mode  
- **Wildcard open** (e.g., `*.bin`) picks the first matching file  

---

## Run

From the project folder:

    python main.py

Then use **File → Open…** and pick any file (you can also type a wildcard like `*.exe`).

---

## Usage & Controls

- **Mode**: switch between *Entropy Map* and *Byte Map* in the left sidebar.  
- **Columns**:
  - Byte Map: number of pixels per row (auto-tuned on load; adjustable).
  - Entropy Map: number of **blocks** per row (image width in blocks).
- **Block size (Entropy Map)**: bytes per entropy block (default 4096).  
- **Palette**: choose Grayscale or Fire.  
- **Zoom**:
  - Mouse Wheel → zoom at cursor  
  - Buttons → `+`, `−`, `Fit`, `1:1`  
  - Double-click image → Fit to window  
- **Status bar**: shows file size, render info, zoom, and subsampling note.  
- **Hover panel**:
  - Byte Map: offset and byte value
  - Entropy Map: block index, byte range, entropy

---

## Performance Notes

- A ~**8,000,000 pixel cap** protects VRAM; very large files will auto-subsample (indicated in the status line).
- **Byte Map** on massive files can still be heavy; try **Entropy Map** with larger **Block size** for faster exploration.
- Switching modes resets the GPU texture to avoid stale data.

---

## Troubleshooting

- **Blank or tiny image** → open a file first; use *Fit to window*.  
- **Slow on huge files** → use Entropy Map with a larger **Block size**, or reduce **Columns**.  
- **VRAM errors** → the 8MP cap should prevent this; if triggered, the status bar will show a message.

---

## License

MIT. See `LICENSE` for details.
