"""Visualization utilities for Ising model simulations."""

from ising_toolkit.visualization.styles import (
    # Style dictionaries
    STYLE_PUBLICATION,
    STYLE_PRESENTATION,
    STYLE_NOTEBOOK,
    STYLE_MINIMAL,
    STYLE_DEFAULT,
    # Style functions
    get_style,
    set_style,
    reset_style,
    style_context,
    list_styles,
    register_style,
    # Color palettes
    COLORS_QUALITATIVE,
    COLORS_TAB10,
    COLORMAP_TEMPERATURE,
    COLORMAP_MAGNETIZATION,
    COLORMAP_SEQUENTIAL,
    COLOR_SPIN_UP,
    COLOR_SPIN_DOWN,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_ORDERED,
    COLOR_DISORDERED,
    COLOR_CRITICAL,
    # Color utilities
    get_colors,
    get_size_colors,
    get_markers,
    # Plotting utilities
    add_critical_line,
    add_theory_line,
    configure_legend,
    format_axis_labels,
)

__all__ = [
    # Style dictionaries
    "STYLE_PUBLICATION",
    "STYLE_PRESENTATION",
    "STYLE_NOTEBOOK",
    "STYLE_MINIMAL",
    "STYLE_DEFAULT",
    # Style functions
    "get_style",
    "set_style",
    "reset_style",
    "style_context",
    "list_styles",
    "register_style",
    # Color palettes
    "COLORS_QUALITATIVE",
    "COLORS_TAB10",
    "COLORMAP_TEMPERATURE",
    "COLORMAP_MAGNETIZATION",
    "COLORMAP_SEQUENTIAL",
    "COLOR_SPIN_UP",
    "COLOR_SPIN_DOWN",
    "COLOR_POSITIVE",
    "COLOR_NEGATIVE",
    "COLOR_ORDERED",
    "COLOR_DISORDERED",
    "COLOR_CRITICAL",
    # Color utilities
    "get_colors",
    "get_size_colors",
    "get_markers",
    # Plotting utilities
    "add_critical_line",
    "add_theory_line",
    "configure_legend",
    "format_axis_labels",
]
