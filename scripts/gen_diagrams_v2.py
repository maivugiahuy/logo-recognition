"""Redraw slides 09, 11, 13, 20 with cleaner layout."""
from PIL import Image, ImageDraw, ImageFont
import math

FONTS = "C:/Windows/Fonts"
OUT = "docs/slides/figures"

# Use system fonts for better glyph coverage
font_title = ImageFont.truetype(f"{FONTS}/arialbd.ttf", 28)
font_box = ImageFont.truetype(f"{FONTS}/arialbd.ttf", 16)
font_box_sm = ImageFont.truetype(f"{FONTS}/arial.ttf", 13)
font_ann = ImageFont.truetype(f"{FONTS}/consola.ttf", 12)
font_footer = ImageFont.truetype(f"{FONTS}/arial.ttf", 11)

BG = "#FFFFFF"
GRAY = (160, 165, 175)
LGRAY = (230, 232, 238)
ANN_C = (130, 135, 148)

PALETTE = {
    "red":    ((255, 237, 234), (215, 82, 72)),
    "orange": ((255, 244, 230), (230, 150, 55)),
    "blue":   ((230, 240, 255), (60, 100, 195)),
    "green":  ((230, 248, 238), (45, 160, 85)),
    "purple": ((244, 234, 255), (135, 75, 195)),
    "gray":   ((245, 246, 250), (175, 178, 190)),
    "teal":   ((228, 248, 248), (40, 150, 150)),
}


def tc(d, text, font, cx, cy, color):
    bb = d.textbbox((0, 0), text, font=font)
    d.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2), text, fill=color, font=font)


def box(d, cx, cy, w, h, pal, label1, label2=None, r=14):
    fill, stroke = PALETTE[pal]
    d.rounded_rectangle((cx - w//2, cy - h//2, cx + w//2, cy + h//2), radius=r, fill=fill, outline=stroke, width=2)
    if label2:
        tc(d, label1, font_box, cx, cy - 10, stroke)
        tc(d, label2, font_box_sm, cx, cy + 10, stroke)
    else:
        tc(d, label1, font_box, cx, cy, stroke)


def sub(d, cx, cy, text):
    tc(d, text, font_footer, cx, cy, ANN_C)


def ann(d, cx, cy, text):
    tc(d, text, font_ann, cx, cy, ANN_C)


def arrow_h(d, x1, y, x2, c=GRAY):
    d.line([(x1, y), (x2 - 8, y)], fill=c, width=2)
    d.polygon([(x2, y), (x2 - 10, y - 5), (x2 - 10, y + 5)], fill=c)


def arrow_v(d, x, y1, y2, c=GRAY):
    if y2 > y1:
        d.line([(x, y1), (x, y2 - 8)], fill=c, width=2)
        d.polygon([(x, y2), (x - 5, y2 - 10), (x + 5, y2 - 10)], fill=c)
    else:
        d.line([(x, y1), (x, y2 + 8)], fill=c, width=2)
        d.polygon([(x, y2), (x - 5, y2 + 10), (x + 5, y2 + 10)], fill=c)


def title_bar(d, W, text):
    tc(d, text, font_title, W // 2, 32, (30, 30, 42))
    d.line([(60, 58), (W - 60, 58)], fill=LGRAY, width=1)


# ═══════════════════════════════════════════════════
# SLIDE 09 — Pipeline
# ═══════════════════════════════════════════════════
def slide09():
    W, H = 1600, 580
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    title_bar(d, W, "Logo Recognition Pipeline")

    y = 230
    bw, bh = 165, 72
    gap = 48
    total = 6 * bw + 5 * gap
    sx = (W - total) // 2 + bw // 2

    stages = [(sx + i * (bw + gap), y) for i in range(6)]
    configs = [
        ("Input Image", None, "gray"),
        ("YOLOv8", "Detection", "red"),
        ("Crop + Resize", "160 x 160", "orange"),
        ("ViT-B/16", "Embedder", "blue"),
        ("FAISS", "Retrieval", "green"),
        ("Threshold", "Decision", "purple"),
    ]
    subs_text = [None, "class-agnostic", None, "CLIP pretrained", "IndexFlatIP", "cosine threshold"]
    anns = [None, "imgsz 512", "logo crops", "160x160", "128-d vector", "top-1 match"]

    for i, ((cx, cy), (l1, l2, pal)) in enumerate(zip(stages, configs)):
        box(d, cx, cy, bw, bh, pal, l1, l2)
        if subs_text[i]:
            sub(d, cx, cy + bh // 2 + 14, subs_text[i])

    for i in range(5):
        x1 = stages[i][0] + bw // 2 + 4
        x2 = stages[i + 1][0] - bw // 2 - 4
        arrow_h(d, x1, y, x2)
        if anns[i + 1]:
            ann(d, (x1 + x2) // 2, y - 48, anns[i + 1])

    # Decision branches
    tcx, tcy = stages[5]
    by = tcy + bh // 2 + 95
    bx_l = tcx - 80
    bx_r = tcx + 80
    bw2, bh2 = 130, 48

    box(d, bx_l, by, bw2, bh2, "green", "Brand Label")
    box(d, bx_r, by, bw2, bh2, "orange", "Unknown")

    fork_y = tcy + bh // 2 + 25
    d.line([(tcx, tcy + bh // 2 + 2), (tcx, fork_y)], fill=GRAY, width=2)
    d.line([(bx_l, fork_y), (bx_r, fork_y)], fill=GRAY, width=2)
    arrow_v(d, bx_l, fork_y, by - bh2 // 2 - 2, PALETTE["green"][1])
    arrow_v(d, bx_r, fork_y, by - bh2 // 2 - 2, PALETTE["orange"][1])

    ann(d, bx_l, fork_y + 22, "score >= 0.50")
    ann(d, bx_r, fork_y + 22, "score < 0.50")

    img.save(f"{OUT}/slide09_pipeline.png", quality=95)
    print("  slide09_pipeline.png")


# ═══════════════════════════════════════════════════
# SLIDE 11 — ViT Architecture
# ═══════════════════════════════════════════════════
def slide11():
    W, H = 1550, 480
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    title_bar(d, W, "ViT-B/16 Embedder Architecture")

    y = 235
    steps = [
        (95,  "gray",   "Input",        "160x160",      "patch grid"),
        (265, "purple", "Patch Embed",  "16x16 conv",   "100 patches"),
        (440, "orange", "+ CLS token",  "+ Pos Embed",  "101 x 768"),
        (645, "blue",   "Transformer",  "x12 blocks",   "12 layers"),
        (835, "green",  "CLS token",    "output",       "512-d (CLIP)"),
        (1000,"red",    "FC Layer",     "512 -> 128",   None),
        (1155,"purple", "L2 Norm",      None,           None),
        (1360,"blue",   "128-d",        "embedding",    "unit vector"),
    ]

    # Draw input patch grid
    ix, iy = 95, y
    for r in range(4):
        for c in range(4):
            x0 = ix - 28 + c * 15
            y0 = iy - 28 + r * 15
            shade = 195 + (r + c) * 8
            d.rectangle([x0, y0, x0 + 13, y0 + 13], fill=(shade, shade + 10, shade + 20), outline=(200, 205, 215))
    sub(d, 95, y + 42, "160x160")

    # Draw remaining boxes
    for cx, pal, l1, l2, note in steps[1:]:
        bw_local = 145
        if pal == "blue" and l1 == "Transformer":
            # Stacked effect
            for k in range(2):
                off = k * 6
                shade_fill = tuple(min(255, c + k * 8) for c in PALETTE[pal][0])
                d.rounded_rectangle((cx - 68 + off, y - 32 + off, cx + 68 + off, y + 32 + off),
                                    radius=10, fill=shade_fill, outline=PALETTE[pal][1], width=1)
            box(d, cx + 12, y + 12, 136, 64, pal, l1, l2)
            if note:
                sub(d, cx + 12, y + 50, note)
        else:
            box(d, cx, y, bw_local, 64, pal, l1, l2)
            if note:
                sub(d, cx, y + 46, note)

    # Arrows
    arrow_positions = [
        (130, 265 - 73),
        (265 + 73, 440 - 73),
        (440 + 73, 645 - 68 - 4),
        (645 + 68 + 14, 835 - 73),
        (835 + 73, 1000 - 73),
        (1000 + 73, 1155 - 73),
        (1155 + 73, 1360 - 73),
    ]
    for x1, x2 in arrow_positions:
        arrow_h(d, x1, y, x2)

    # Footer
    tc(d, "CLIP OpenAI pretrained  |  Pos embed bicubic 224->160  |  All 12 blocks fine-tuned",
       font_footer, W // 2, H - 28, (190, 192, 200))

    img.save(f"{OUT}/slide11_vit_architecture.png", quality=95)
    print("  slide11_vit_architecture.png")


# ═══════════════════════════════════════════════════
# SLIDE 13 — Two-Phase Training
# ═══════════════════════════════════════════════════
def slide13():
    W, H = 1500, 600
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    title_bar(d, W, "Two-Phase Training Pipeline")

    bw, bh = 180, 64
    gap = 50

    # ── Row 1: Phase A (top) ──
    y1 = 145
    xa1 = 130                          # Open-set data
    xa2 = xa1 + bw + gap               # Phase A
    xa3 = xa2 + bw + gap               # base.pt
    xa4 = xa3 + bw + gap + 30          # HN Mining
    xa5 = xa4 + bw + gap + 10          # hn_map.json

    box(d, xa1, y1, bw, bh, "gray", "Open-set", "1,582 classes")
    box(d, xa2, y1, bw, bh, "blue", "Phase A", "ArcFace K=1")
    box(d, xa3, y1, bw, bh, "green", "base.pt", "checkpoint")
    box(d, xa4, y1, bw + 20, bh, "orange", "HN Mining", "confusion matrix")
    box(d, xa5, y1, bw, bh, "orange", "hn_map.json", "HN pairs")

    sub(d, xa1, y1 + bh // 2 + 16, "MPerClassSampler")
    sub(d, xa2, y1 + bh // 2 + 16, "base training")
    sub(d, xa4, y1 + bh // 2 + 16, "+ Levenshtein filter")

    arrow_h(d, xa1 + bw // 2 + 4, y1, xa2 - bw // 2 - 4)
    arrow_h(d, xa2 + bw // 2 + 4, y1, xa3 - bw // 2 - 4)
    arrow_h(d, xa3 + bw // 2 + 4, y1, xa4 - (bw + 20) // 2 - 4)
    arrow_h(d, xa4 + (bw + 20) // 2 + 4, y1, xa5 - bw // 2 - 4)

    # Row 1 label
    tc(d, "PHASE A", font_ann, 48, y1, PALETTE["blue"][1])

    # ── Separator line ──
    y_sep = 260
    d.line([(60, y_sep), (W - 60, y_sep)], fill=(235, 237, 242), width=1)

    # ── Row 2: Phase C (bottom) ──
    y2 = 370
    xb1 = 130                          # Closed-set data
    xb2 = xb1 + bw + gap              # Phase C
    xb3 = xb2 + bw + gap              # hn.pt

    box(d, xb1, y2, bw, bh, "gray", "Closed-set", "1,977 classes")
    box(d, xb2, y2, bw, bh, "red", "Phase C", "Sub-center K=3")
    box(d, xb3, y2, bw, bh, "green", "hn.pt", "final checkpoint")

    sub(d, xb1, y2 + bh // 2 + 16, "HardNegBatchSampler")
    sub(d, xb2, y2 + bh // 2 + 16, "HN training")

    arrow_h(d, xb1 + bw // 2 + 4, y2, xb2 - bw // 2 - 4)
    arrow_h(d, xb2 + bw // 2 + 4, y2, xb3 - bw // 2 - 4)

    # Row 2 label
    tc(d, "PHASE C", font_ann, 48, y2, PALETTE["red"][1])

    # ── Vertical connections ──
    seg = 6
    blue = PALETTE["blue"][1]
    orange = PALETTE["orange"][1]

    # 1) base.pt -> Phase C: "init weights" (dotted blue, down then left)
    # Go down from base.pt, bend left to Phase C top
    init_x1 = xa3                       # start x (base.pt)
    init_x2 = xb2                       # end x (Phase C)
    init_y1 = y1 + bh // 2 + 4         # start y
    bend_y = y_sep + 20                 # bend point y
    init_y2 = y2 - bh // 2 - 4         # end y (top of Phase C)

    # vertical dotted segment down
    y_cur = init_y1
    while y_cur < bend_y:
        y_end = min(y_cur + seg, bend_y)
        d.line([(init_x1, y_cur), (init_x1, y_end)], fill=blue, width=2)
        y_cur += seg * 2
    # horizontal dotted segment left
    x_cur = init_x1
    while x_cur > init_x2:
        x_end = max(x_cur - seg, init_x2)
        d.line([(x_cur, bend_y), (x_end, bend_y)], fill=blue, width=2)
        x_cur -= seg * 2
    # vertical dotted segment down to Phase C
    y_cur = bend_y
    while y_cur < init_y2 - 12:
        y_end = min(y_cur + seg, init_y2 - 12)
        d.line([(init_x2, y_cur), (init_x2, y_end)], fill=blue, width=2)
        y_cur += seg * 2
    # arrowhead pointing down into Phase C
    d.polygon([(init_x2, init_y2), (init_x2 - 5, init_y2 - 10), (init_x2 + 5, init_y2 - 10)],
              fill=blue)
    tc(d, "init weights", font_ann, (init_x1 + init_x2) // 2, bend_y - 14, blue)

    # 2) hn_map.json -> Closed-set sampler: feeds sampler (dotted orange, down then left)
    hn_x1 = xa5                         # start x (hn_map.json)
    hn_x2 = xb1                         # end x (Closed-set / sampler)
    hn_y1 = y1 + bh // 2 + 4           # start y
    hn_bend_y = y_sep + 48             # bend point y (below init weights line)
    hn_y2 = y2 - bh // 2 - 4          # end y

    # vertical dotted down
    y_cur = hn_y1
    while y_cur < hn_bend_y:
        y_end = min(y_cur + seg, hn_bend_y)
        d.line([(hn_x1, y_cur), (hn_x1, y_end)], fill=orange, width=2)
        y_cur += seg * 2
    # horizontal dotted left
    x_cur = hn_x1
    while x_cur > hn_x2:
        x_end = max(x_cur - seg, hn_x2)
        d.line([(x_cur, hn_bend_y), (x_end, hn_bend_y)], fill=orange, width=2)
        x_cur -= seg * 2
    # vertical dotted down into Closed-set
    y_cur = hn_bend_y
    while y_cur < hn_y2 - 12:
        y_end = min(y_cur + seg, hn_y2 - 12)
        d.line([(hn_x2, y_cur), (hn_x2, y_end)], fill=orange, width=2)
        y_cur += seg * 2
    # arrowhead
    d.polygon([(hn_x2, hn_y2), (hn_x2 - 5, hn_y2 - 10), (hn_x2 + 5, hn_y2 - 10)],
              fill=orange)
    tc(d, "hn_map.json", font_ann, (hn_x1 + hn_x2) // 2, hn_bend_y - 14, orange)

    # ── Footer ──
    tc(d, "Phase A trains base embeddings on open-set classes  |  HN Mining finds confusable pairs  |  Phase C refines with hard negatives",
       font_footer, W // 2, H - 24, (190, 192, 200))

    img.save(f"{OUT}/slide13_training_flow.png", quality=95)
    print("  slide13_training_flow.png")


# ═══════════════════════════════════════════════════
# SLIDE 20 — Ensemble
# ═══════════════════════════════════════════════════
def slide20():
    W, H = 1400, 520
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    title_bar(d, W, "Ensemble Pipeline: ViT-B/16 + DINOv3")

    bw, bh = 155, 58
    y_mid = 265
    y_top = 170
    y_bot = 360

    # Input + Detector
    in_x = 100
    det_x = 290
    split_x = 420

    # Branches
    emb_x = 580
    gal_x = 770

    # Fusion + Output
    fus_x = 970
    out_x = 1200

    # Input
    box(d, in_x, y_mid, 120, bh, "gray", "Input Image")
    arrow_h(d, in_x + 62, y_mid, det_x - 78)

    # Detector
    box(d, det_x, y_mid, 155, bh, "red", "YOLOv8", "Detection")
    sub(d, det_x, y_mid + bh // 2 + 12, "class-agnostic, 1 class")

    # Fork
    d.line([(det_x + 78, y_mid), (split_x, y_mid)], fill=GRAY, width=2)
    d.line([(split_x, y_top), (split_x, y_bot)], fill=GRAY, width=2)
    ann(d, split_x + 40, y_mid - 14, "crops")

    # ViT branch (top)
    arrow_h(d, split_x, y_top, emb_x - 78, PALETTE["blue"][1])
    box(d, emb_x, y_top, bw, bh, "blue", "ViT-B/16", "Embedder")
    sub(d, emb_x, y_top + bh // 2 + 12, "CLIP pretrained")

    arrow_h(d, emb_x + 78, y_top, gal_x - 78, PALETTE["blue"][1])
    box(d, gal_x, y_top, bw, bh, "blue", "ViT Gallery", "FAISS top-20")

    # DINOv3 branch (bottom)
    arrow_h(d, split_x, y_bot, emb_x - 78, PALETTE["purple"][1])
    box(d, emb_x, y_bot, bw, bh, "purple", "DINOv3-B/16", "Embedder")
    sub(d, emb_x, y_bot + bh // 2 + 12, "self-supervised")

    arrow_h(d, emb_x + 78, y_bot, gal_x - 78, PALETTE["purple"][1])
    box(d, gal_x, y_bot, bw, bh, "purple", "DINOv3 Gallery", "FAISS top-20")

    # Converge lines
    merge_x = fus_x - 82
    d.line([(gal_x + 78, y_top), (merge_x, y_top)], fill=PALETTE["blue"][1], width=2)
    d.line([(merge_x, y_top), (merge_x, y_mid)], fill=GRAY, width=2)
    d.line([(gal_x + 78, y_bot), (merge_x, y_bot)], fill=PALETTE["purple"][1], width=2)
    d.line([(merge_x, y_bot), (merge_x, y_mid)], fill=GRAY, width=2)
    arrow_h(d, merge_x, y_mid, fus_x - 68)

    # Fusion
    box(d, fus_x, y_mid, 135, 62, "orange", "Score Fusion")
    ann(d, fus_x, y_mid + 46, "w * vit + (1-w) * dino")

    # Output
    arrow_h(d, fus_x + 68, y_mid, out_x - 90)
    box(d, out_x, y_mid, 165, bh, "green", "Brand Label /", "Unknown")

    img.save(f"{OUT}/slide20_ensemble.png", quality=95)
    print("  slide20_ensemble.png")


if __name__ == "__main__":
    print("Redrawing diagrams v2...\n")
    slide09()
    slide11()
    slide13()
    slide20()
    print("\nDone!")
