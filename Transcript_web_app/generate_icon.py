#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


OUT = Path("assets/icon_1024.png")
SIZE = 1024


img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# Rounded square background
pad = 56
d.rounded_rectangle(
    (pad, pad, SIZE - pad, SIZE - pad),
    radius=220,
    fill=(26, 78, 137, 255),
)

# Inner card
d.rounded_rectangle(
    (240, 200, 784, 824),
    radius=56,
    fill=(244, 248, 255, 255),
)

# Text lines
for y in (300, 380, 460):
    d.rounded_rectangle((300, y, 710, y + 28), radius=14, fill=(166, 186, 213, 255))

# Subtitle bubble
d.rounded_rectangle((300, 600, 710, 720), radius=24, fill=(52, 112, 184, 255))
d.rounded_rectangle((346, 642, 664, 674), radius=12, fill=(236, 244, 255, 255))

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(f"Wrote {OUT}")
