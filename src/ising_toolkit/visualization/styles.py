"""Publication-quality matplotlib styles for Ising model visualizations."""

from contextlib import contextmanager
from typing import Dict, List, Optional

# =============================================================================
# Style Dictionaries
# =============================================================================

STYLE_PUBLICATION = {
    # Figure settings
    'figure.figsize': (6, 4.5),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'figure.autolayout': True,

    # Font settings (serif for publications)
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',  # Computer Modern for math

    # Axes labels and titles
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'axes.labelpad': 4.0,

    # Legend
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'legend.fancybox': False,
    'legend.borderpad': 0.4,
    'legend.labelspacing': 0.4,
    'legend.handlelength': 1.5,

    # Ticks
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,

    # Lines and markers
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'lines.markeredgewidth': 0.8,

    # Axes
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.spines.top': True,
    'axes.spines.right': True,

    # Grid
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.linestyle': '-',

    # Error bars
    'errorbar.capsize': 3,

    # Saving
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
}


STYLE_PRESENTATION = {
    # Figure settings (larger for slides)
    'figure.figsize': (10, 7),
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'figure.autolayout': True,

    # Font settings (sans-serif for presentations)
    'font.size': 16,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',

    # Axes labels and titles (larger)
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'bold',
    'axes.labelpad': 6.0,

    # Legend (larger)
    'legend.fontsize': 14,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'legend.fancybox': True,
    'legend.borderpad': 0.5,
    'legend.labelspacing': 0.5,
    'legend.handlelength': 2.0,

    # Ticks (larger)
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.top': False,
    'ytick.right': False,

    # Lines and markers (thicker)
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
    'lines.markeredgewidth': 1.5,

    # Axes
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Grid
    'grid.alpha': 0.4,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',

    # Error bars
    'errorbar.capsize': 5,

    # Saving
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
}


STYLE_NOTEBOOK = {
    # Figure settings (good for Jupyter)
    'figure.figsize': (8, 6),
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',

    # Font settings
    'font.size': 12,
    'font.family': 'sans-serif',
    'mathtext.fontset': 'dejavusans',

    # Axes
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.grid': True,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Legend
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.framealpha': 0.8,

    # Ticks
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    # Lines
    'lines.linewidth': 2.0,
    'lines.markersize': 8,

    # Grid
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,

    # Saving
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
}


STYLE_MINIMAL = {
    # Clean, minimal style
    'figure.figsize': (6, 4),
    'figure.dpi': 100,
    'figure.facecolor': 'white',

    'font.size': 11,
    'font.family': 'sans-serif',

    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.grid': False,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,

    'legend.fontsize': 10,
    'legend.frameon': False,

    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'lines.linewidth': 1.5,
    'lines.markersize': 6,

    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
}


STYLE_DEFAULT = {
    # Matplotlib defaults (reset)
    'figure.figsize': (6.4, 4.8),
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'figure.autolayout': False,

    'font.size': 10,
    'font.family': 'sans-serif',
    'mathtext.fontset': 'dejavusans',

    'axes.labelsize': 'medium',
    'axes.titlesize': 'large',
    'axes.grid': False,
    'axes.linewidth': 0.8,
    'axes.spines.top': True,
    'axes.spines.right': True,

    'legend.fontsize': 'medium',
    'legend.frameon': True,
    'legend.framealpha': 0.8,

    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.top': False,
    'ytick.right': False,

    'lines.linewidth': 1.5,
    'lines.markersize': 6,

    'grid.alpha': 1.0,
    'grid.linewidth': 0.8,

    'savefig.dpi': 'figure',
    'savefig.bbox': None,
}


# Registry of available styles
_STYLE_REGISTRY: Dict[str, dict] = {
    'publication': STYLE_PUBLICATION,
    'presentation': STYLE_PRESENTATION,
    'notebook': STYLE_NOTEBOOK,
    'minimal': STYLE_MINIMAL,
    'default': STYLE_DEFAULT,
}


# =============================================================================
# Color Palettes
# =============================================================================

# Qualitative colors for distinguishing different system sizes, algorithms, etc.
# Based on ColorBrewer Set1 + additional distinct colors
COLORS_QUALITATIVE = [
    '#377eb8',  # Blue
    '#e41a1c',  # Red
    '#4daf4a',  # Green
    '#984ea3',  # Purple
    '#ff7f00',  # Orange
    '#a65628',  # Brown
    '#f781bf',  # Pink
    '#999999',  # Gray
    '#17becf',  # Cyan
    '#bcbd22',  # Olive
]

# Alternative qualitative palette (tab10)
COLORS_TAB10 = [
    '#1f77b4',  # tab:blue
    '#ff7f0e',  # tab:orange
    '#2ca02c',  # tab:green
    '#d62728',  # tab:red
    '#9467bd',  # tab:purple
    '#8c564b',  # tab:brown
    '#e377c2',  # tab:pink
    '#7f7f7f',  # tab:gray
    '#bcbd22',  # tab:olive
    '#17becf',  # tab:cyan
]

# Sequential colormaps for temperature
COLORMAP_TEMPERATURE = 'coolwarm'  # Cold (blue) to hot (red)
COLORMAP_TEMPERATURE_ALT = 'RdYlBu_r'  # Red-Yellow-Blue reversed
COLORMAP_VIRIDIS = 'viridis'  # Perceptually uniform

# Diverging colormap for magnetization (-1 to +1)
COLORMAP_MAGNETIZATION = 'RdBu'  # Red (down) - White (0) - Blue (up)
COLORMAP_MAGNETIZATION_ALT = 'coolwarm'

# Sequential colormap for energy or other quantities
COLORMAP_SEQUENTIAL = 'plasma'
COLORMAP_SEQUENTIAL_ALT = 'inferno'

# Spin colors
COLOR_SPIN_UP = '#1f77b4'     # tab:blue
COLOR_SPIN_DOWN = '#d62728'   # tab:red
COLOR_POSITIVE = '#1f77b4'    # Alias for spin up
COLOR_NEGATIVE = '#d62728'    # Alias for spin down

# Phase colors
COLOR_ORDERED = '#2ca02c'     # Green for T < Tc
COLOR_DISORDERED = '#ff7f0e'  # Orange for T > Tc
COLOR_CRITICAL = '#9467bd'    # Purple for T = Tc

# Additional useful colors
COLOR_DATA = '#1f77b4'        # Default data color
COLOR_FIT = '#d62728'         # Fit line color
COLOR_THEORY = '#2ca02c'      # Theoretical prediction
COLOR_ERROR = '#999999'       # Error bands/bars
COLOR_HIGHLIGHT = '#ff7f0e'   # Highlighting


# =============================================================================
# Marker styles for different data series
# =============================================================================

MARKERS_SIZES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
MARKERS_DEFAULT = ['o', 's', '^', 'D', 'v']


# =============================================================================
# Style Functions
# =============================================================================

def get_style(style: str = 'publication') -> dict:
    """Get a style dictionary.

    Parameters
    ----------
    style : str
        Style name. Options: 'publication', 'presentation', 'notebook',
        'minimal', 'default'.

    Returns
    -------
    dict
        Dictionary of matplotlib rcParams.

    Examples
    --------
    >>> style = get_style('publication')
    >>> plt.rcParams.update(style)
    """
    style_lower = style.lower()

    if style_lower not in _STYLE_REGISTRY:
        available = list(_STYLE_REGISTRY.keys())
        raise ValueError(
            f"Unknown style: '{style}'. Available: {available}"
        )

    return _STYLE_REGISTRY[style_lower].copy()


def set_style(style: str = 'publication') -> None:
    """Apply a style globally.

    Parameters
    ----------
    style : str
        Style name. Options: 'publication', 'presentation', 'notebook',
        'minimal', 'default'.

    Examples
    --------
    >>> from ising_toolkit.visualization import set_style
    >>> set_style('publication')
    >>> # All subsequent plots will use publication style
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required. Install with: pip install matplotlib"
        )

    style_dict = get_style(style)
    plt.rcParams.update(style_dict)


def reset_style() -> None:
    """Reset matplotlib to default style.

    Examples
    --------
    >>> set_style('publication')
    >>> # ... make plots ...
    >>> reset_style()  # Back to defaults
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required")

    plt.rcdefaults()


@contextmanager
def style_context(style: str = 'publication'):
    """Context manager for temporary style changes.

    Parameters
    ----------
    style : str
        Style name to apply temporarily.

    Yields
    ------
    None

    Examples
    --------
    >>> with style_context('publication'):
    ...     plt.plot(x, y)
    ...     plt.savefig('figure.pdf')
    >>> # Style is restored after the block
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required")

    style_dict = get_style(style)

    with plt.rc_context(style_dict):
        yield


def get_colors(n: int, palette: str = 'qualitative') -> List[str]:
    """Get a list of colors for plotting.

    Parameters
    ----------
    n : int
        Number of colors needed.
    palette : str
        Color palette to use:
        - 'qualitative': Distinct colors for categories
        - 'tab10': Matplotlib tab10 palette
        - 'sequential': Sequential colormap samples

    Returns
    -------
    list of str
        List of hex color strings.

    Examples
    --------
    >>> colors = get_colors(4)
    >>> for i, L in enumerate([8, 16, 32, 64]):
    ...     plt.plot(T, M[L], color=colors[i], label=f'L={L}')
    """
    if palette == 'qualitative':
        colors = COLORS_QUALITATIVE
    elif palette == 'tab10':
        colors = COLORS_TAB10
    elif palette == 'sequential':
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(COLORMAP_SEQUENTIAL)
            return [cmap(i / max(n - 1, 1)) for i in range(n)]
        except ImportError:
            colors = COLORS_QUALITATIVE
    else:
        colors = COLORS_QUALITATIVE

    # Cycle if we need more colors
    if n <= len(colors):
        return colors[:n]
    else:
        return [colors[i % len(colors)] for i in range(n)]


def get_size_colors(sizes: List[int]) -> Dict[int, str]:
    """Get color mapping for system sizes.

    Parameters
    ----------
    sizes : list of int
        System sizes.

    Returns
    -------
    dict
        Mapping from size to color.

    Examples
    --------
    >>> size_colors = get_size_colors([8, 16, 32, 64])
    >>> for L, color in size_colors.items():
    ...     plt.plot(T, U[L], color=color, label=f'L={L}')
    """
    colors = get_colors(len(sizes))
    return {size: color for size, color in zip(sorted(sizes), colors)}


def get_markers(n: int) -> List[str]:
    """Get a list of marker styles.

    Parameters
    ----------
    n : int
        Number of markers needed.

    Returns
    -------
    list of str
        List of marker style strings.
    """
    markers = MARKERS_SIZES
    if n <= len(markers):
        return markers[:n]
    else:
        return [markers[i % len(markers)] for i in range(n)]


# =============================================================================
# Plotting utility functions
# =============================================================================

def add_critical_line(
    ax,
    Tc: float,
    orientation: str = 'vertical',
    label: Optional[str] = None,
    **kwargs,
):
    """Add a line marking the critical temperature.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add line to.
    Tc : float
        Critical temperature.
    orientation : str
        'vertical' (default) or 'horizontal'.
    label : str, optional
        Label for legend.
    **kwargs
        Additional arguments for axvline/axhline.

    Returns
    -------
    line
        The line artist.
    """
    defaults = {
        'color': COLOR_CRITICAL,
        'linestyle': '--',
        'linewidth': 1.5,
        'alpha': 0.7,
        'zorder': 0,
    }
    defaults.update(kwargs)

    if label is None:
        label = f'$T_c = {Tc:.3f}$'

    if orientation == 'vertical':
        return ax.axvline(Tc, label=label, **defaults)
    else:
        return ax.axhline(Tc, label=label, **defaults)


def add_theory_line(
    ax,
    x,
    y,
    label: str = 'Theory',
    **kwargs,
):
    """Add a theoretical prediction line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add line to.
    x, y : array-like
        Data for the line.
    label : str
        Label for legend.
    **kwargs
        Additional arguments for plot.

    Returns
    -------
    line
        The line artist.
    """
    defaults = {
        'color': COLOR_THEORY,
        'linestyle': '-',
        'linewidth': 1.5,
        'zorder': 5,
    }
    defaults.update(kwargs)

    return ax.plot(x, y, label=label, **defaults)


def configure_legend(
    ax,
    location: str = 'best',
    **kwargs,
):
    """Configure legend with good defaults.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to configure.
    location : str
        Legend location.
    **kwargs
        Additional arguments for legend.
    """
    defaults = {
        'loc': location,
        'frameon': True,
        'framealpha': 0.9,
        'edgecolor': '0.8',
    }
    defaults.update(kwargs)

    ax.legend(**defaults)


def format_axis_labels(
    ax,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
):
    """Set axis labels with proper formatting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to configure.
    xlabel, ylabel : str, optional
        Axis labels.
    title : str, optional
        Plot title.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


# =============================================================================
# Register available styles
# =============================================================================

def list_styles() -> List[str]:
    """List available style names.

    Returns
    -------
    list of str
        Available style names.
    """
    return list(_STYLE_REGISTRY.keys())


def register_style(name: str, style_dict: dict) -> None:
    """Register a custom style.

    Parameters
    ----------
    name : str
        Name for the style.
    style_dict : dict
        Dictionary of rcParams.

    Examples
    --------
    >>> my_style = {'font.size': 14, 'axes.grid': True}
    >>> register_style('my_custom', my_style)
    >>> set_style('my_custom')
    """
    _STYLE_REGISTRY[name.lower()] = style_dict.copy()
