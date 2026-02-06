"""Phase diagram and observable plotting functions."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
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
    COLORMAP_TEMPERATURE,
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
    ax.set_ylabel(f'$L^{{{exponent:.2f}}} \\cdot$ {_OBSERVABLE_SHORT_LABELS.get(observable, observable)}')
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
