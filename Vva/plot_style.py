"""
Publication-quality plot styling configuration.

Sets up matplotlib to produce LaTeX-compatible, clean, professional plots
with STIX fonts and no grids.
"""

import matplotlib.pyplot as plt


def setup_publication_style():
    """
    Configure matplotlib for LaTeX-compatible, publication-ready plots.

    Features:
    - STIX fonts (LaTeX-like, widely available)
    - Matching text and math fonts
    - No grids (clean look)
    - Closed box axes (all four spines visible)
    - Reasonable font sizes (not too large)
    - Thinner line widths for axes
    """
    plt.rcParams.update({
        # Font configuration - STIX (LaTeX-like, widely available)
        'font.family': 'STIXGeneral',
        'mathtext.fontset': 'stix',

        # Font sizes - reasonable and consistent
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 13,

        # Axes styling - clean and minimal, closed box
        'axes.linewidth': 0.75,
        'axes.edgecolor': '.8',
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,

        # Tick marks - hidden (labels kept, marks removed)
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,

        # Legend - no frame
        'legend.frameon': False,

        # Figure defaults
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
