""""Old Money" visual identity for every Part A figure.

Understated, classic, editorial: ink/navy and muted brass on a warm off-white
page, serif type, a light warm-grey grid, and no top/right spines. One shared
SECTOR_COLORS map keeps every sector-coloured figure consistent.

Usage:
    from src import plotstyle
    plotstyle.apply_style()          # once, before drawing
    ...
    plotstyle.caption(fig, "Figure 1. ... units, sample 2020-2023.")

The palette is deliberately muted, so it trades some of a dashboard palette's
raw colour separation for the editorial look. Identity is therefore never
carried by colour alone: every figure keeps a legend and labels key points.
"""
import matplotlib as mpl

# ── Core palette (each hex named) ─────────────────────────────────────────────
INK = "#1F2A44"        # primary ink / navy - text, primary marks
BRASS = "#B08D57"      # muted brass/gold - the single accent
HUNTER = "#3A5A40"     # hunter green - supporting
SLATE = "#46535C"      # slate grey - supporting
BURGUNDY = "#6E2A34"   # burgundy - supporting
PAPER = "#FAF7F0"      # warm off-white background (page)
GRID = "#E4DCCB"       # light warm-grey grid
INK_MUTED = "#6B6353"  # muted warm grey-brown - secondary text / captions

# ── One shared sector -> colour map (10 equity sectors + crypto) ──────────────
# Muted and harmonious; spread across hue AND lightness so neighbours separate
# (worst normal-vision pair OKLab dE ~6 on the dataviz validator - as far apart
# as 11 low-chroma tones reach). Keys match the sector strings in the data.
SECTOR_COLORS = {
    "Tech": INK,            # #1F2A44 navy
    "Financials": HUNTER,   # #3A5A40 hunter green
    "Energy": BURGUNDY,     # #6E2A34 burgundy
    "Consumer": BRASS,      # #B08D57 brass
    "Industrials": SLATE,   # #46535C slate
    "Healthcare": "#7FA1B0",  # dusty blue
    "Comm": "#9C6B4F",        # clay
    "Materials": "#948BA0",   # heather grey-purple
    "Utilities": "#90A17C",   # sage green
    "RealEstate": "#8C5B6E",  # mauve
}
CRYPTO_COLOR = "#C2703D"   # burnt orange - crypto, distinct from every sector

# Line styles let same-family crypto series stay distinct without extra hues.
CRYPTO_LINESTYLES = ["-", "--", ":", "-."]


def sector_color(sector: str) -> str:
    """Colour for a sector name; falls back to muted ink for anything unknown."""
    return SECTOR_COLORS.get(sector, INK_MUTED)


def apply_style() -> None:
    """Set matplotlib rcParams to the Old Money identity. Call once."""
    mpl.rcParams.update({
        # Type: classic serif, restrained sizes.
        "font.family": "serif",
        "font.serif": ["Georgia", "Palatino Linotype", "Palatino",
                       "Book Antiqua", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 15,
        # Surfaces: warm off-white page.
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        # Ink for text and axes.
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SLATE,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titlecolor": INK,
        # Recessive warm-grey grid; hide top/right spines.
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        # Marks and output.
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.autolayout": True,
    })


def caption(fig, text: str) -> None:
    """Add a small italic caption line at the bottom of a figure.

    Save with ``bbox_inches="tight"`` so the caption is not clipped.
    """
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=7.5,
             style="italic", color=INK_MUTED, wrap=True)
