"""Phase diagram and observable plotting functions."""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # noqa: F401
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ising_toolkit.visualization.styles import (
    style_context,
    get_size_colors,
    get_markers,
    add_critical_line,
    configure_legend,
    COLOR_CRITICAL,
)


# Observable display names and error column mappings
_OBSERVABLE_LABELS = {
    'energy_mean': r'Energy per spin ($\langle E \rangle / N$)',
    'magnetization_mean': r'Magnetization per spin ($\langle M \rangle / N$)',
    'abs_magnetization_mean': r'$|\langle M \rangle| / N$',
    'heat_capacity': r'Heat capacity ($C / N$)',
    'susceptibility': r'Susceptibility ($\chi / N$)',
    'binder_cumulant': r'Binder cumulant ($U$)',
}

_OBSERVABLE_SHORT_LABELS = {
    'energy_mean': 'Energy',
    'magnetization_mean': 'Magnetization',
    'abs_magnetization_mean': '|Magnetization|',
    'heat_capacity': 'Heat capacity',
    'susceptibility': 'Susceptibility',
    'binder_cumulant': 'Binder cumulant',
}

_ERROR_COLUMNS = {
    'energy_mean': 'energy_err',
    'magnetization_mean': 'magnetization_err',
    'abs_magnetization_mean': 'abs_magnetization_err',
}


def _get_Tc_from_data(data) -> Optional[float]:
    """Try to detect critical temperature from data."""
    try:
        from ising_toolkit.utils.constants import CRITICAL_TEMP_2D, CRITICAL_TEMP_3D

        # Check temperature range
        temps = data['temperature'].values

        # If temperatures span around 2D Tc, assume 2D
        if temps.min() < CRITICAL_TEMP_2D < temps.max():
            return CRITICAL_TEMP_2D

        # If temperatures span around 3D Tc, assume 3D
        if temps.min() < CRITICAL_TEMP_3D < temps.max():
            return CRITICAL_TEMP_3D

    except Exception:
        pass

    return None


def _filter_sizes(data, sizes: Optional[List[int]] = None):
    """Filter data by sizes if specified."""
    if sizes is None:
        return data

    if 'size' not in data.columns:
        return data

    return data[data['size'].isin(sizes)]


def plot_observable_vs_temperature(
    data,
    observable: str,
    sizes: Optional[List[int]] = None,
    ax=None,
    show_errors: bool = True,
    show_Tc: bool = True,
    Tc: Optional[float] = None,
    colors: Optional[Dict[int, str]] = None,
    markers: Optional[List[str]] = None,
    label_prefix: str = 'L = ',
    **kwargs,
):
    """Plot single observable vs temperature.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'temperature' and observable columns.
        If 'size' column present, plots separate curves for each size.
    observable : str
        Column name to plot (e.g., 'heat_capacity', 'susceptibility').
    sizes : list of int, optional
        Filter to specific sizes. Default is all sizes.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_errors : bool, optional
        Show error bars if error column exists. Default is True.
    show_Tc : bool, optional
        Show vertical line at critical temperature. Default is True.
    Tc : float, optional
        Critical temperature. Auto-detected if None.
    colors : dict, optional
        Mapping from size to color. Auto-generated if None.
    markers : list, optional
        List of marker styles. Auto-generated if None.
    label_prefix : str, optional
        Prefix for legend labels. Default is 'L = '.
    **kwargs
        Additional arguments passed to plot/errorbar.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> ax = plot_observable_vs_temperature(df, 'susceptibility')
    >>> ax = plot_observable_vs_temperature(
    ...     df, 'heat_capacity',
    ...     sizes=[16, 32, 64],
    ...     show_Tc=True
    ... )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if not HAS_PANDAS:
        raise ImportError("pandas required for plotting")

    # Filter by sizes
    data = _filter_sizes(data, sizes)

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots()

    # Check if we have multiple sizes
    has_sizes = 'size' in data.columns

    if has_sizes:
        unique_sizes = sorted(data['size'].unique())

        # Get colors and markers
        if colors is None:
            colors = get_size_colors(unique_sizes)
        if markers is None:
            markers = get_markers(len(unique_sizes))

        # Plot each size
        for i, size in enumerate(unique_sizes):
            size_data = data[data['size'] == size].sort_values('temperature')
            temps = size_data['temperature'].values
            values = size_data[observable].values

            color = colors.get(size, colors.get(i, None))
            marker = markers[i] if i < len(markers) else 'o'
            label = f'{label_prefix}{size}'

            # Check for error column
            err_col = _ERROR_COLUMNS.get(observable)
            if show_errors and err_col and err_col in size_data.columns:
                errors = size_data[err_col].values
                ax.errorbar(
                    temps, values, yerr=errors,
                    fmt=f'{marker}-',
                    color=color,
                    label=label,
                    capsize=3,
                    markersize=5,
                    **kwargs
                )
            else:
                ax.plot(
                    temps, values,
                    f'{marker}-',
                    color=color,
                    label=label,
                    markersize=5,
                    **kwargs
                )
    else:
        # Single dataset
        data = data.sort_values('temperature')
        temps = data['temperature'].values
        values = data[observable].values

        err_col = _ERROR_COLUMNS.get(observable)
        if show_errors and err_col and err_col in data.columns:
            errors = data[err_col].values
            ax.errorbar(
                temps, values, yerr=errors,
                fmt='o-',
                capsize=3,
                **kwargs
            )
        else:
            ax.plot(temps, values, 'o-', **kwargs)

    # Labels
    ylabel = _OBSERVABLE_LABELS.get(observable, observable)
    ax.set_xlabel('Temperature ($T$)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # Show critical temperature
    if show_Tc:
        if Tc is None:
            Tc = _get_Tc_from_data(data)
        if Tc is not None:
            add_critical_line(ax, Tc)

    # Legend
    if has_sizes:
        configure_legend(ax)

    return ax


def plot_phase_diagram(
    data,
    observables: Optional[List[str]] = None,
    sizes: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    style: str = 'publication',
    show_Tc: bool = True,
    Tc: Optional[float] = None,
    save: Optional[str] = None,
):
    """Create multi-panel phase diagram figure.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with temperature and observable columns.
    observables : list of str, optional
        Observables to plot. Default is:
        ['abs_magnetization_mean', 'energy_mean', 'heat_capacity', 'susceptibility'].
    sizes : list of int, optional
        Filter to specific sizes.
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    style : str, optional
        Matplotlib style to use. Default is 'publication'.
    show_Tc : bool, optional
        Show critical temperature lines. Default is True.
    Tc : float, optional
        Critical temperature. Auto-detected if None.
    save : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray
        Array of axes objects.

    Examples
    --------
    >>> fig, axes = plot_phase_diagram(df)
    >>> fig, axes = plot_phase_diagram(
    ...     df,
    ...     observables=['heat_capacity', 'susceptibility'],
    ...     sizes=[16, 32, 64],
    ...     save='phase_diagram.pdf'
    ... )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if not HAS_PANDAS:
        raise ImportError("pandas required for plotting")

    # Default observables
    if observables is None:
        observables = [
            'abs_magnetization_mean',
            'energy_mean',
            'heat_capacity',
            'susceptibility',
        ]

    # Filter available observables
    observables = [obs for obs in observables if obs in data.columns]

    if len(observables) == 0:
        raise ValueError("No valid observables found in data")

    # Determine layout
    n_obs = len(observables)
    if n_obs == 1:
        n_rows, n_cols = 1, 1
    elif n_obs == 2:
        n_rows, n_cols = 1, 2
    elif n_obs <= 4:
        n_rows, n_cols = 2, 2
    elif n_obs <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_cols = 3
        n_rows = (n_obs + 2) // 3

    # Figure size
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    # Get colors for sizes
    if 'size' in data.columns:
        unique_sizes = sorted(data['size'].unique())
        if sizes is not None:
            unique_sizes = [s for s in unique_sizes if s in sizes]
        colors = get_size_colors(unique_sizes)
        markers = get_markers(len(unique_sizes))
    else:
        colors = None
        markers = None

    # Create figure with style
    with style_context(style):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        if n_obs == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each observable
        for i, obs in enumerate(observables):
            plot_observable_vs_temperature(
                data,
                obs,
                sizes=sizes,
                ax=axes[i],
                show_Tc=show_Tc,
                Tc=Tc,
                colors=colors,
                markers=markers,
            )

            # Use shorter label for subplots
            axes[i].set_ylabel(_OBSERVABLE_SHORT_LABELS.get(obs, obs))

        # Hide unused axes
        for i in range(len(observables), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')

    return fig, axes[:len(observables)]


def plot_binder_cumulant(
    data,
    sizes: Optional[List[int]] = None,
    ax=None,
    show_crossing: bool = True,
    show_reference_lines: bool = True,
    Tc: Optional[float] = None,
    save: Optional[str] = None,
    **kwargs,
):
    """Plot Binder cumulant vs temperature for multiple sizes.

    The Binder cumulant U = 1 - <m^4>/(3<m^2>^2) is useful for
    locating phase transitions. Curves for different sizes cross
    at the critical temperature.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'temperature', 'binder_cumulant', and optionally 'size'.
    sizes : list of int, optional
        Filter to specific sizes.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    show_crossing : bool, optional
        Highlight the crossing point. Default is True.
    show_reference_lines : bool, optional
        Show U = 2/3 and U = 0 reference lines. Default is True.
    Tc : float, optional
        Critical temperature. Auto-detected if None.
    save : str, optional
        Path to save figure.
    **kwargs
        Additional arguments passed to plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> ax = plot_binder_cumulant(df, sizes=[8, 16, 32, 64])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if not HAS_PANDAS:
        raise ImportError("pandas required for plotting")

    # Filter by sizes
    data = _filter_sizes(data, sizes)

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Check if we have multiple sizes
    has_sizes = 'size' in data.columns

    if has_sizes:
        unique_sizes = sorted(data['size'].unique())
        colors = get_size_colors(unique_sizes)
        markers = get_markers(len(unique_sizes))

        for i, size in enumerate(unique_sizes):
            size_data = data[data['size'] == size].sort_values('temperature')
            temps = size_data['temperature'].values
            U = size_data['binder_cumulant'].values

            ax.plot(
                temps, U,
                f'{markers[i]}-',
                color=colors[size],
                label=f'L = {size}',
                markersize=5,
                **kwargs
            )
    else:
        data = data.sort_values('temperature')
        ax.plot(
            data['temperature'],
            data['binder_cumulant'],
            'o-',
            **kwargs
        )

    # Reference lines
    if show_reference_lines:
        ax.axhline(2/3, color='gray', linestyle=':', alpha=0.5, label=r'$U = 2/3$ (ordered)')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5, label=r'$U = 0$ (disordered)')

    # Find and show crossing
    if show_crossing and has_sizes and len(unique_sizes) >= 2:
        crossing_T, crossing_U = _find_binder_crossing(data, unique_sizes)
        if crossing_T is not None:
            ax.axvline(
                crossing_T,
                color=COLOR_CRITICAL,
                linestyle='--',
                alpha=0.7,
                label=f'$T_c \\approx {crossing_T:.3f}$'
            )
            ax.scatter(
                [crossing_T], [crossing_U],
                s=100, c=COLOR_CRITICAL, marker='*',
                zorder=10, edgecolors='black', linewidths=0.5
            )

    # Labels
    ax.set_xlabel('Temperature ($T$)')
    ax.set_ylabel('Binder cumulant ($U$)')
    ax.grid(True, alpha=0.3)

    # Legend
    configure_legend(ax, location='best')

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    return ax


def _find_binder_crossing(data, sizes: List[int]) -> Tuple[Optional[float], Optional[float]]:
    """Find approximate Binder cumulant crossing point."""
    try:
        crossings_T = []
        crossings_U = []

        for i, L1 in enumerate(sizes[:-1]):
            for L2 in sizes[i + 1:]:
                data1 = data[data['size'] == L1].sort_values('temperature')
                data2 = data[data['size'] == L2].sort_values('temperature')

                temps1 = data1['temperature'].values
                U1 = data1['binder_cumulant'].values
                temps2 = data2['temperature'].values
                U2 = data2['binder_cumulant'].values

                # Interpolate to common grid
                temp_min = max(temps1.min(), temps2.min())
                temp_max = min(temps1.max(), temps2.max())
                temp_grid = np.linspace(temp_min, temp_max, 200)

                U1_interp = np.interp(temp_grid, temps1, U1)
                U2_interp = np.interp(temp_grid, temps2, U2)

                # Find crossing
                diff = U1_interp - U2_interp
                sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

                if len(sign_changes) > 0:
                    idx = sign_changes[0]
                    t1, t2 = temp_grid[idx], temp_grid[idx + 1]
                    d1, d2 = diff[idx], diff[idx + 1]
                    Tc = t1 - d1 * (t2 - t1) / (d2 - d1)
                    Uc = np.interp(Tc, temp_grid, U1_interp)
                    crossings_T.append(Tc)
                    crossings_U.append(Uc)

        if crossings_T:
            return float(np.mean(crossings_T)), float(np.mean(crossings_U))

    except Exception:
        pass

    return None, None


def plot_scaling_collapse(
    data,
    observable: str,
    Tc: float,
    nu: float,
    exponent: float,
    sizes: Optional[List[int]] = None,
    ax=None,
    save: Optional[str] = None,
    **kwargs,
):
    """Plot data collapse for finite-size scaling.

    Plots L^exponent * observable vs (T - Tc) * L^(1/nu).
    Good scaling collapse indicates correct Tc and exponents.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'temperature', observable, and 'size' columns.
    observable : str
        Observable to collapse.
    Tc : float
        Critical temperature.
    nu : float
        Correlation length exponent.
    exponent : float
        Scaling exponent (beta/nu for magnetization, -gamma/nu for chi).
    sizes : list of int, optional
        Filter to specific sizes.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    save : str, optional
        Path to save figure.
    **kwargs
        Additional arguments passed to plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if not HAS_PANDAS:
        raise ImportError("pandas required for plotting")

    data = _filter_sizes(data, sizes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if 'size' not in data.columns:
        raise ValueError("Data must have 'size' column for scaling collapse")

    unique_sizes = sorted(data['size'].unique())
    colors = get_size_colors(unique_sizes)
    markers = get_markers(len(unique_sizes))

    for i, size in enumerate(unique_sizes):
        size_data = data[data['size'] == size].sort_values('temperature')
        temps = size_data['temperature'].values
        values = size_data[observable].values

        # Scaling variables
        x = (temps - Tc) * (size ** (1 / nu))
        y = values * (size ** exponent)

        ax.plot(
            x, y,
            markers[i],
            color=colors[size],
            label=f'L = {size}',
            markersize=6,
            **kwargs
        )

    ax.set_xlabel(r'$(T - T_c) \cdot L^{1/\nu}$')
    obs_label = _OBSERVABLE_SHORT_LABELS.get(observable, observable)
    ax.set_ylabel(f'$L^{{{exponent:.2f}}} \\cdot$ {obs_label}')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    configure_legend(ax)

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    return ax


def plot_configuration(
    spins: np.ndarray,
    ax=None,
    cmap: str = 'RdBu',
    show_colorbar: bool = True,
    title: Optional[str] = None,
    save: Optional[str] = None,
):
    """Plot 2D spin configuration.

    Parameters
    ----------
    spins : np.ndarray
        2D array of spins (+1 or -1).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, optional
        Colormap. Default is 'RdBu' (red for -1, blue for +1).
    show_colorbar : bool, optional
        Show colorbar. Default is True.
    title : str, optional
        Plot title.
    save : str, optional
        Path to save figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(spins, cmap=cmap, vmin=-1, vmax=1, origin='lower')

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['-1 (down)', '0', '+1 (up)'])

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if title:
        ax.set_title(title)

    ax.set_aspect('equal')

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return ax


def plot_energy_histogram(
    energy: np.ndarray,
    ax=None,
    bins: int = 50,
    density: bool = True,
    show_mean: bool = True,
    title: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs,
):
    """Plot histogram of energy values.

    Parameters
    ----------
    energy : np.ndarray
        Energy time series.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    bins : int, optional
        Number of bins. Default is 50.
    density : bool, optional
        Normalize to probability density. Default is True.
    show_mean : bool, optional
        Show vertical line at mean. Default is True.
    title : str, optional
        Plot title.
    save : str, optional
        Path to save figure.
    **kwargs
        Additional arguments passed to hist.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(energy, bins=bins, density=density, alpha=0.7, **kwargs)

    if show_mean:
        mean_E = np.mean(energy)
        ax.axvline(mean_E, color='red', linestyle='--',
                   label=f'Mean = {mean_E:.2f}')
        ax.legend()

    ax.set_xlabel('Energy')
    ax.set_ylabel('Probability density' if density else 'Count')

    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return ax


def plot_time_series(
    data: np.ndarray,
    observable_name: str = 'Observable',
    ax=None,
    show_mean: bool = True,
    show_std: bool = True,
    save: Optional[str] = None,
    **kwargs,
):
    """Plot time series of an observable.

    Parameters
    ----------
    data : np.ndarray
        Time series data.
    observable_name : str, optional
        Name for y-axis label.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    show_mean : bool, optional
        Show horizontal line at mean. Default is True.
    show_std : bool, optional
        Show shaded region for ±1 std. Default is True.
    save : str, optional
        Path to save figure.
    **kwargs
        Additional arguments passed to plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    steps = np.arange(len(data))
    ax.plot(steps, data, linewidth=0.5, alpha=0.8, **kwargs)

    mean = np.mean(data)
    std = np.std(data)

    if show_mean:
        ax.axhline(mean, color='red', linestyle='--',
                   label=f'Mean = {mean:.4f}')

    if show_std:
        ax.axhspan(mean - std, mean + std, alpha=0.2, color='red',
                   label=f'±1σ = {std:.4f}')

    ax.set_xlabel('Monte Carlo step')
    ax.set_ylabel(observable_name)
    ax.grid(True, alpha=0.3)

    if show_mean or show_std:
        ax.legend(loc='upper right')

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return ax


def plot_spin_configuration(
    spins: np.ndarray,
    ax=None,
    cmap: str = 'RdBu',
    title: Optional[str] = None,
    show_colorbar: bool = False,
    save: Optional[str] = None,
    **kwargs,
):
    """Plot 2D spin configuration as heatmap.

    Parameters
    ----------
    spins : np.ndarray
        2D array with values +1 or -1.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    cmap : str, optional
        Colormap. Default is 'RdBu' (red for -1, blue for +1).
    title : str, optional
        Plot title.
    show_colorbar : bool, optional
        Show colorbar. Default is False.
    save : str, optional
        Path to save figure.
    **kwargs
        Additional arguments passed to imshow.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> spins = model.spins
    >>> ax = plot_spin_configuration(spins, title='T = 2.0')
    >>> ax = plot_spin_configuration(spins, cmap='coolwarm', show_colorbar=True)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if spins.ndim != 2:
        raise ValueError(f"Expected 2D array, got {spins.ndim}D")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        spins,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin='lower',
        interpolation='nearest',
        **kwargs
    )

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['-1', '0', '+1'])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return ax


def plot_configuration_comparison(
    configurations: List[np.ndarray],
    titles: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'RdBu',
    show_colorbar: bool = True,
    save: Optional[str] = None,
):
    """Plot multiple configurations side by side.

    Useful for comparing spin configurations at different temperatures:
    T < Tc (ordered), T = Tc (critical), T > Tc (disordered).

    Parameters
    ----------
    configurations : list of np.ndarray
        List of 2D spin configurations.
    titles : list of str, optional
        Titles for each subplot. Default is None.
    figsize : tuple, optional
        Figure size. Default is (4 * n_configs, 4).
    cmap : str, optional
        Colormap. Default is 'RdBu'.
    show_colorbar : bool, optional
        Show colorbar on last panel. Default is True.
    save : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray
        Array of axes objects.

    Examples
    --------
    >>> configs = [spins_low_T, spins_Tc, spins_high_T]
    >>> titles = ['T = 1.5 (ordered)', 'T = 2.27 (critical)', 'T = 3.0 (disordered)']
    >>> fig, axes = plot_configuration_comparison(configs, titles=titles)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    n_configs = len(configurations)
    if n_configs == 0:
        raise ValueError("At least one configuration required")

    if titles is not None and len(titles) != n_configs:
        raise ValueError(
            f"Number of titles ({len(titles)}) must match "
            f"configurations ({n_configs})"
        )

    if figsize is None:
        figsize = (4 * n_configs, 4)

    fig, axes = plt.subplots(1, n_configs, figsize=figsize)

    if n_configs == 1:
        axes = np.array([axes])

    for i, (config, ax) in enumerate(zip(configurations, axes)):
        if config.ndim != 2:
            raise ValueError(f"Configuration {i} must be 2D, got {config.ndim}D")

        im = ax.imshow(
            config,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            origin='lower',
            interpolation='nearest',
        )

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        if titles is not None:
            ax.set_title(titles[i])

    # Add colorbar to the last axes
    if show_colorbar:
        cbar = fig.colorbar(im, ax=axes[-1], shrink=0.8)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['↓', '', '↑'])

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_spin_configuration_3d(
    spins: np.ndarray,
    ax=None,
    slice_axis: int = 2,
    slice_index: Optional[int] = None,
    cmap: str = 'RdBu',
    title: Optional[str] = None,
    show_colorbar: bool = False,
    save: Optional[str] = None,
    **kwargs,
):
    """Plot slice of 3D spin configuration.

    Extracts a 2D slice from a 3D configuration and plots it.

    Parameters
    ----------
    spins : np.ndarray
        3D array with values +1 or -1.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    slice_axis : int, optional
        Axis perpendicular to the slice (0, 1, or 2). Default is 2 (z-axis).
    slice_index : int, optional
        Index along slice_axis. Default is middle of array.
    cmap : str, optional
        Colormap. Default is 'RdBu'.
    title : str, optional
        Plot title. If None, generates automatic title.
    show_colorbar : bool, optional
        Show colorbar. Default is False.
    save : str, optional
        Path to save figure.
    **kwargs
        Additional arguments passed to imshow.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> # Plot xy-plane at z=L/2
    >>> ax = plot_spin_configuration_3d(spins_3d, slice_axis=2)
    >>> # Plot xz-plane at y=0
    >>> ax = plot_spin_configuration_3d(spins_3d, slice_axis=1, slice_index=0)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if spins.ndim != 3:
        raise ValueError(f"Expected 3D array, got {spins.ndim}D")

    if slice_axis not in [0, 1, 2]:
        raise ValueError(f"slice_axis must be 0, 1, or 2, got {slice_axis}")

    # Default to middle slice
    if slice_index is None:
        slice_index = spins.shape[slice_axis] // 2

    if slice_index < 0 or slice_index >= spins.shape[slice_axis]:
        raise ValueError(
            f"slice_index {slice_index} out of bounds for axis {slice_axis} "
            f"with size {spins.shape[slice_axis]}"
        )

    # Extract slice
    if slice_axis == 0:
        slice_2d = spins[slice_index, :, :]
        axis_labels = ('y', 'z')
        axis_name = 'x'
    elif slice_axis == 1:
        slice_2d = spins[:, slice_index, :]
        axis_labels = ('x', 'z')
        axis_name = 'y'
    else:  # slice_axis == 2
        slice_2d = spins[:, :, slice_index]
        axis_labels = ('x', 'y')
        axis_name = 'z'

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        slice_2d,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin='lower',
        interpolation='nearest',
        **kwargs
    )

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['-1', '0', '+1'])

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect('equal')

    if title is None:
        title = f'{axis_name} = {slice_index}'
    ax.set_title(title)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return ax


def plot_spin_configuration_3d_slices(
    spins: np.ndarray,
    n_slices: int = 4,
    slice_axis: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'RdBu',
    save: Optional[str] = None,
):
    """Plot multiple slices of 3D spin configuration.

    Creates a grid showing evenly-spaced slices through a 3D configuration.

    Parameters
    ----------
    spins : np.ndarray
        3D array with values +1 or -1.
    n_slices : int, optional
        Number of slices to show. Default is 4.
    slice_axis : int, optional
        Axis perpendicular to the slices (0, 1, or 2). Default is 2 (z-axis).
    figsize : tuple, optional
        Figure size. Default is (4 * n_slices, 4).
    cmap : str, optional
        Colormap. Default is 'RdBu'.
    save : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray
        Array of axes objects.

    Examples
    --------
    >>> fig, axes = plot_spin_configuration_3d_slices(spins_3d, n_slices=5)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if spins.ndim != 3:
        raise ValueError(f"Expected 3D array, got {spins.ndim}D")

    if slice_axis not in [0, 1, 2]:
        raise ValueError(f"slice_axis must be 0, 1, or 2, got {slice_axis}")

    axis_size = spins.shape[slice_axis]
    n_slices = min(n_slices, axis_size)

    # Calculate slice indices
    slice_indices = np.linspace(0, axis_size - 1, n_slices, dtype=int)

    if figsize is None:
        figsize = (4 * n_slices, 4)

    fig, axes = plt.subplots(1, n_slices, figsize=figsize)

    if n_slices == 1:
        axes = np.array([axes])

    axis_names = ['x', 'y', 'z']
    axis_name = axis_names[slice_axis]

    for ax, idx in zip(axes, slice_indices):
        plot_spin_configuration_3d(
            spins,
            ax=ax,
            slice_axis=slice_axis,
            slice_index=idx,
            cmap=cmap,
            title=f'{axis_name} = {idx}',
            show_colorbar=False,
        )

    # Add colorbar to the figure
    fig.colorbar(
        ax.images[0],
        ax=axes.tolist(),
        shrink=0.6,
        location='right',
        label='Spin',
        ticks=[-1, 0, 1],
    )

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return fig, axes


# ============================================================================
# Analysis-specific plots
# ============================================================================

def plot_autocorrelation(
    data: np.ndarray,
    max_lag: Optional[int] = None,
    ax=None,
    show_tau: bool = True,
    show_confidence: bool = True,
    save: Optional[str] = None,
    **kwargs,
):
    """Plot autocorrelation function with integrated time annotation.

    Parameters
    ----------
    data : np.ndarray
        Time series data.
    max_lag : int, optional
        Maximum lag to plot. Default is min(len(data)//4, 500).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_tau : bool, optional
        Show integrated autocorrelation time annotation. Default is True.
    show_confidence : bool, optional
        Show 95% confidence interval. Default is True.
    save : str, optional
        Path to save figure.
    **kwargs
        Additional arguments passed to plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> energy = sampler.run(10000)['energy']
    >>> ax = plot_autocorrelation(energy, max_lag=100)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    n = len(data)
    if max_lag is None:
        max_lag = min(n // 4, 500)

    # Compute autocorrelation function
    mean = np.mean(data)
    var = np.var(data)

    if var == 0:
        raise ValueError("Data has zero variance, cannot compute autocorrelation")

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        cov = np.mean((data[:-lag] - mean) * (data[lag:] - mean))
        acf[lag] = cov / var

    lags = np.arange(max_lag + 1)

    # Plot ACF
    ax.plot(lags, acf, 'b-', linewidth=1.5, label='ACF', **kwargs)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)

    # Confidence interval (95% for white noise)
    if show_confidence:
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci, color='red', linestyle='--', alpha=0.5, label=f'95% CI (±{ci:.3f})')
        ax.axhline(-ci, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(lags, -ci, ci, color='red', alpha=0.1)

    # Compute and show integrated autocorrelation time
    if show_tau:
        tau_int = _compute_tau_int(acf)
        ax.axvline(tau_int, color='green', linestyle=':', linewidth=2,
                   label=f'τ_int ≈ {tau_int:.1f}')

        # Shade the integration region
        integration_mask = lags <= tau_int * 6  # Show ~6τ
        ax.fill_between(
            lags[integration_mask],
            0,
            acf[integration_mask],
            alpha=0.2,
            color='blue'
        )

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation C(t)')
    ax.set_title('Autocorrelation Function')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Set reasonable y-limits
    ax.set_ylim(-0.2, 1.1)
    ax.set_xlim(0, max_lag)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return ax


def _compute_tau_int(acf: np.ndarray, c: float = 5.0) -> float:
    """Compute integrated autocorrelation time using Sokal's windowing.

    Parameters
    ----------
    acf : np.ndarray
        Autocorrelation function values.
    c : float
        Window coefficient (typically 4-6).

    Returns
    -------
    float
        Integrated autocorrelation time.
    """
    tau_int = 0.5  # Start with C(0)/2 = 0.5
    max_lag = len(acf) - 1

    for t in range(1, max_lag + 1):
        if acf[t] <= 0:
            break
        tau_int += acf[t]
        # Sokal's automatic windowing
        if t >= c * tau_int:
            break

    return tau_int


def plot_blocking_analysis(
    data: np.ndarray,
    ax=None,
    show_plateau: bool = True,
    max_block_size: Optional[int] = None,
    save: Optional[str] = None,
    **kwargs,
):
    """Plot standard error vs block size from blocking analysis.

    The blocking method transforms correlated data into independent blocks.
    The standard error plateaus when the block size exceeds the correlation
    length.

    Parameters
    ----------
    data : np.ndarray
        Time series data.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_plateau : bool, optional
        Estimate and show plateau value. Default is True.
    max_block_size : int, optional
        Maximum block size to analyze. Default is len(data)//4.
    save : str, optional
        Path to save figure.
    **kwargs
        Additional arguments passed to plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> energy = sampler.run(10000)['energy']
    >>> ax = plot_blocking_analysis(energy)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    n = len(data)
    if max_block_size is None:
        max_block_size = n // 4

    # Compute blocked standard errors
    block_sizes = []
    std_errors = []

    # Use logarithmically spaced block sizes for efficiency
    sizes = np.unique(np.logspace(0, np.log10(max_block_size), 50).astype(int))

    for block_size in sizes:
        if block_size > max_block_size:
            break

        n_blocks = n // block_size
        if n_blocks < 2:
            break

        # Compute block means
        blocked_data = data[:n_blocks * block_size].reshape(n_blocks, block_size)
        block_means = np.mean(blocked_data, axis=1)

        # Standard error of block means
        se = np.std(block_means, ddof=1) / np.sqrt(n_blocks)

        block_sizes.append(block_size)
        std_errors.append(se)

    block_sizes = np.array(block_sizes)
    std_errors = np.array(std_errors)

    # Plot
    ax.loglog(block_sizes, std_errors, 'bo-', markersize=4, linewidth=1, **kwargs)

    # Naive standard error (no blocking)
    naive_se = np.std(data, ddof=1) / np.sqrt(n)
    ax.axhline(naive_se, color='gray', linestyle=':', alpha=0.7,
               label=f'Naive σ/√n = {naive_se:.4f}')

    # Estimate and show plateau
    if show_plateau and len(std_errors) > 5:
        # Use the last few points as plateau estimate
        plateau_estimate = np.mean(std_errors[-5:])
        ax.axhline(plateau_estimate, color='red', linestyle='--',
                   label=f'Plateau ≈ {plateau_estimate:.4f}')

        # Estimate effective sample size
        if naive_se > 0:
            tau_eff = (plateau_estimate / naive_se) ** 2
            ax.text(
                0.95, 0.95,
                f'τ_eff ≈ {tau_eff:.1f}',
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

    ax.set_xlabel('Block size')
    ax.set_ylabel('Standard error')
    ax.set_title('Blocking Analysis')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, which='both')

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return ax


def plot_correlation_time_comparison(
    data: np.ndarray,
    ax=None,
    figsize: Tuple[float, float] = (12, 5),
    save: Optional[str] = None,
):
    """Plot autocorrelation and blocking analysis side by side.

    Useful for comparing different methods of estimating correlation time.

    Parameters
    ----------
    data : np.ndarray
        Time series data.
    ax : tuple of axes, optional
        Tuple of (ax1, ax2). If None, creates new figure.
    figsize : tuple, optional
        Figure size. Default is (12, 5).
    save : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : tuple
        Tuple of axes (ax_acf, ax_blocking).

    Examples
    --------
    >>> fig, axes = plot_correlation_time_comparison(energy_data)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        ax1, ax2 = ax
        fig = ax1.figure

    plot_autocorrelation(data, ax=ax1)
    plot_blocking_analysis(data, ax=ax2)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return fig, (ax1, ax2)


def plot_equilibration_check(
    data: np.ndarray,
    observable_name: str = 'Observable',
    window_size: Optional[int] = None,
    ax=None,
    save: Optional[str] = None,
):
    """Plot time series with running mean to check equilibration.

    Parameters
    ----------
    data : np.ndarray
        Time series data.
    observable_name : str, optional
        Name for y-axis label.
    window_size : int, optional
        Window size for running mean. Default is len(data)//20.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    save : str, optional
        Path to save figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    n = len(data)
    if window_size is None:
        window_size = max(n // 20, 10)

    steps = np.arange(n)

    # Raw data
    ax.plot(steps, data, 'b-', linewidth=0.3, alpha=0.5, label='Raw data')

    # Running mean
    running_mean = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    running_steps = steps[window_size//2:window_size//2 + len(running_mean)]
    ax.plot(running_steps, running_mean, 'r-', linewidth=2,
            label=f'Running mean (window={window_size})')

    # Overall mean (after equilibration estimate)
    # Use last 80% as equilibrated
    equilibrated = data[n//5:]
    mean_eq = np.mean(equilibrated)
    std_eq = np.std(equilibrated)

    ax.axhline(mean_eq, color='green', linestyle='--',
               label=f'Mean (last 80%) = {mean_eq:.4f}')
    ax.axhspan(mean_eq - std_eq, mean_eq + std_eq, alpha=0.1, color='green')

    # Mark suggested equilibration cutoff
    ax.axvline(n//5, color='orange', linestyle=':', linewidth=2,
               label='Suggested equilibration')

    ax.set_xlabel('Monte Carlo step')
    ax.set_ylabel(observable_name)
    ax.set_title('Equilibration Check')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return ax


def plot_bootstrap_distribution(
    data: np.ndarray,
    statistic: str = 'mean',
    n_bootstrap: int = 1000,
    ax=None,
    show_ci: bool = True,
    save: Optional[str] = None,
):
    """Plot bootstrap distribution of a statistic.

    Parameters
    ----------
    data : np.ndarray
        Original data.
    statistic : str, optional
        Statistic to bootstrap ('mean', 'std', 'var'). Default is 'mean'.
    n_bootstrap : int, optional
        Number of bootstrap samples. Default is 1000.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    show_ci : bool, optional
        Show 95% confidence interval. Default is True.
    save : str, optional
        Path to save figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    n = len(data)

    # Define statistic function
    stat_funcs = {
        'mean': np.mean,
        'std': np.std,
        'var': np.var,
    }

    if statistic not in stat_funcs:
        raise ValueError(f"Unknown statistic: {statistic}. Use one of {list(stat_funcs.keys())}")

    stat_func = stat_funcs[statistic]

    # Bootstrap
    np.random.seed(42)  # For reproducibility
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = stat_func(sample)

    # Plot histogram
    ax.hist(bootstrap_stats, bins=50, density=True, alpha=0.7,
            edgecolor='black', linewidth=0.5)

    # Original statistic
    original_stat = stat_func(data)
    ax.axvline(original_stat, color='red', linestyle='--', linewidth=2,
               label=f'Original {statistic} = {original_stat:.4f}')

    # Confidence interval
    if show_ci:
        ci_low = np.percentile(bootstrap_stats, 2.5)
        ci_high = np.percentile(bootstrap_stats, 97.5)
        ax.axvline(ci_low, color='green', linestyle=':', linewidth=2)
        ax.axvline(ci_high, color='green', linestyle=':', linewidth=2,
                   label=f'95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
        ax.axvspan(ci_low, ci_high, alpha=0.2, color='green')

    ax.set_xlabel(f'{statistic.capitalize()}')
    ax.set_ylabel('Probability density')
    ax.set_title(f'Bootstrap Distribution of {statistic.capitalize()} (n={n_bootstrap})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return ax


def plot_finite_size_scaling(
    sizes: List[int],
    observables: Dict[int, float],
    errors: Optional[Dict[int, float]] = None,
    exponent: Optional[float] = None,
    observable_name: str = 'Observable',
    ax=None,
    log_scale: bool = True,
    show_fit: bool = True,
    save: Optional[str] = None,
):
    """Plot observable vs system size for finite-size scaling.

    Parameters
    ----------
    sizes : list of int
        System sizes.
    observables : dict
        Dictionary mapping size to observable value at Tc.
    errors : dict, optional
        Dictionary mapping size to error.
    exponent : float, optional
        Expected scaling exponent. If given, shows theoretical line.
    observable_name : str, optional
        Name for y-axis label.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    log_scale : bool, optional
        Use log-log scale. Default is True.
    show_fit : bool, optional
        Show power law fit. Default is True.
    save : str, optional
        Path to save figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> sizes = [8, 16, 32, 64]
    >>> chi_max = {8: 10.2, 16: 25.1, 32: 61.5, 64: 148.3}
    >>> plot_finite_size_scaling(sizes, chi_max, exponent=1.75)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sizes = np.array(sizes)
    obs_values = np.array([observables[L] for L in sizes])

    # Get errors if available
    if errors is not None:
        err_values = np.array([errors.get(L, 0) for L in sizes])
        has_errors = True
    else:
        err_values = None
        has_errors = False

    # Plot data
    if has_errors and np.any(err_values > 0):
        ax.errorbar(sizes, obs_values, yerr=err_values, fmt='o',
                    markersize=8, capsize=4, label='Data')
    else:
        ax.plot(sizes, obs_values, 'o', markersize=8, label='Data')

    # Power law fit
    if show_fit and len(sizes) >= 3:
        log_L = np.log(sizes)
        log_obs = np.log(obs_values)

        # Linear fit in log space
        coeffs = np.polyfit(log_L, log_obs, 1)
        fitted_exponent = coeffs[0]
        amplitude = np.exp(coeffs[1])

        # Plot fit line
        L_fit = np.linspace(sizes.min() * 0.8, sizes.max() * 1.2, 100)
        obs_fit = amplitude * L_fit ** fitted_exponent

        ax.plot(L_fit, obs_fit, 'r--', linewidth=2,
                label=f'Fit: L^{fitted_exponent:.3f}')

    # Theoretical exponent line
    if exponent is not None:
        # Scale to pass through first data point
        amplitude_theory = obs_values[0] / (sizes[0] ** exponent)
        L_theory = np.linspace(sizes.min() * 0.8, sizes.max() * 1.2, 100)
        obs_theory = amplitude_theory * L_theory ** exponent

        ax.plot(L_theory, obs_theory, 'g:', linewidth=2,
                label=f'Theory: L^{exponent:.3f}')

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel('System size L')
    ax.set_ylabel(observable_name)
    ax.set_title('Finite-Size Scaling')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')

    return ax
