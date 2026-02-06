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

from ising_toolkit.visualization.plots import (
    # Phase diagram plotting
    plot_observable_vs_temperature,
    plot_phase_diagram,
    plot_binder_cumulant,
    plot_scaling_collapse,
    # Configuration plotting
    plot_configuration,
    plot_spin_configuration,
    plot_configuration_comparison,
    plot_spin_configuration_3d,
    plot_spin_configuration_3d_slices,
    # Histogram and time series
    plot_energy_histogram,
    plot_time_series,
)

from ising_toolkit.visualization.animation import (
    create_spin_animation,
    create_observable_animation,
    create_temperature_sweep_animation,
    create_domain_growth_animation,
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
    # Phase diagram plotting
    "plot_observable_vs_temperature",
    "plot_phase_diagram",
    "plot_binder_cumulant",
    "plot_scaling_collapse",
    # Configuration plotting
    "plot_configuration",
    "plot_spin_configuration",
    "plot_configuration_comparison",
    "plot_spin_configuration_3d",
    "plot_spin_configuration_3d_slices",
    # Histogram and time series
    "plot_energy_histogram",
    "plot_time_series",
    # Animation
    "create_spin_animation",
    "create_observable_animation",
    "create_temperature_sweep_animation",
    "create_domain_growth_animation",
]
