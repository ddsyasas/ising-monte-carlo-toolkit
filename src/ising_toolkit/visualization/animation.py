"""Animation utilities for Ising model spin evolution."""

from typing import List, Optional, Tuple

import numpy as np


def create_spin_animation(
    configurations: List[np.ndarray],
    filename: str,
    fps: int = 10,
    cmap: str = 'RdBu',
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 100,
    title_template: str = 'Step {step}',
    add_colorbar: bool = False,
    interval: Optional[int] = None,
) -> None:
    """Create animation of spin configuration evolution.

    Parameters
    ----------
    configurations : list of np.ndarray
        List of 2D spin configurations to animate.
    filename : str
        Output file path. Use .gif for GIF format or .mp4 for video.
    fps : int, optional
        Frames per second. Default is 10.
    cmap : str, optional
        Colormap for spins. Default is 'RdBu' (red=-1, blue=+1).
    figsize : tuple, optional
        Figure size in inches. Default is (6, 6).
    dpi : int, optional
        Resolution in dots per inch. Default is 100.
    title_template : str, optional
        Format string for frame titles. Receives 'step' keyword.
        Default is 'Step {step}'.
    add_colorbar : bool, optional
        Add colorbar to animation. Default is False.
    interval : int, optional
        Delay between frames in milliseconds. If None, computed from fps.

    Raises
    ------
    ValueError
        If no configurations provided or configurations are not 2D.
    ImportError
        If required dependencies are not available.

    Examples
    --------
    >>> # Collect configurations during simulation
    >>> configs = []
    >>> for i in range(100):
    ...     sampler.step()
    ...     if i % 10 == 0:
    ...         configs.append(model.spins.copy())
    >>> create_spin_animation(configs, 'evolution.gif', fps=5)

    >>> # With custom title showing temperature
    >>> create_spin_animation(
    ...     configs, 'evolution.mp4',
    ...     title_template='T=2.27, Step {step}'
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        raise ImportError("matplotlib required for animation")

    if len(configurations) == 0:
        raise ValueError("At least one configuration required")

    # Validate configurations
    for i, config in enumerate(configurations):
        if config.ndim != 2:
            raise ValueError(
                f"Configuration {i} must be 2D, got {config.ndim}D"
            )

    if interval is None:
        interval = 1000 // fps  # milliseconds per frame

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Initial plot
    im = ax.imshow(
        configurations[0],
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin='lower',
        interpolation='nearest',
    )

    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['↓', '', '↑'])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    title = ax.set_title(title_template.format(step=0))

    def update(frame):
        """Update function for animation."""
        im.set_array(configurations[frame])
        title.set_text(title_template.format(step=frame))
        return [im, title]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(configurations),
        interval=interval,
        blit=True,
    )

    # Save animation
    _save_animation(anim, filename, fps, dpi)

    plt.close(fig)


def create_observable_animation(
    times: np.ndarray,
    observable: np.ndarray,
    configurations: List[np.ndarray],
    filename: str,
    observable_name: str = 'Magnetization',
    fps: int = 10,
    cmap: str = 'RdBu',
    figsize: Tuple[float, float] = (12, 5),
    dpi: int = 100,
    interval: Optional[int] = None,
) -> None:
    """Create side-by-side animation of spins and observable time series.

    Left panel shows the spin configuration, right panel shows the
    observable time series with a moving marker indicating current time.

    Parameters
    ----------
    times : np.ndarray
        Time values (e.g., Monte Carlo steps) for observable.
    observable : np.ndarray
        Observable values corresponding to times.
    configurations : list of np.ndarray
        List of 2D spin configurations. Should have same length as times.
    filename : str
        Output file path. Use .gif for GIF format or .mp4 for video.
    observable_name : str, optional
        Name for y-axis label. Default is 'Magnetization'.
    fps : int, optional
        Frames per second. Default is 10.
    cmap : str, optional
        Colormap for spins. Default is 'RdBu'.
    figsize : tuple, optional
        Figure size in inches. Default is (12, 5).
    dpi : int, optional
        Resolution in dots per inch. Default is 100.
    interval : int, optional
        Delay between frames in milliseconds. If None, computed from fps.

    Raises
    ------
    ValueError
        If array lengths don't match or configurations are invalid.

    Examples
    --------
    >>> # Collect data during simulation
    >>> configs = []
    >>> magnetizations = []
    >>> steps = []
    >>> for i in range(100):
    ...     sampler.step()
    ...     if i % 10 == 0:
    ...         configs.append(model.spins.copy())
    ...         magnetizations.append(np.mean(model.spins))
    ...         steps.append(i)
    >>> create_observable_animation(
    ...     np.array(steps),
    ...     np.array(magnetizations),
    ...     configs,
    ...     'simulation.gif'
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        raise ImportError("matplotlib required for animation")

    # Validate inputs
    if len(configurations) == 0:
        raise ValueError("At least one configuration required")

    if len(times) != len(observable):
        raise ValueError(
            f"times ({len(times)}) and observable ({len(observable)}) "
            "must have same length"
        )

    if len(configurations) != len(times):
        raise ValueError(
            f"configurations ({len(configurations)}) and times ({len(times)}) "
            "must have same length"
        )

    for i, config in enumerate(configurations):
        if config.ndim != 2:
            raise ValueError(
                f"Configuration {i} must be 2D, got {config.ndim}D"
            )

    if interval is None:
        interval = 1000 // fps

    # Create figure with two panels
    fig, (ax_spin, ax_obs) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: spin configuration
    im = ax_spin.imshow(
        configurations[0],
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin='lower',
        interpolation='nearest',
    )
    ax_spin.set_xlabel('x')
    ax_spin.set_ylabel('y')
    ax_spin.set_aspect('equal')
    spin_title = ax_spin.set_title(f'Step {int(times[0])}')

    # Right panel: observable time series
    ax_obs.plot(times, observable, 'b-', alpha=0.7, linewidth=1)
    marker, = ax_obs.plot(
        [times[0]], [observable[0]],
        'ro', markersize=10, zorder=10
    )
    vline = ax_obs.axvline(times[0], color='red', linestyle='--', alpha=0.5)

    ax_obs.set_xlabel('Monte Carlo step')
    ax_obs.set_ylabel(observable_name)
    ax_obs.grid(True, alpha=0.3)

    # Add mean line
    mean_val = np.mean(observable)
    ax_obs.axhline(mean_val, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()

    def update(frame):
        """Update function for animation."""
        # Update spin configuration
        im.set_array(configurations[frame])
        spin_title.set_text(f'Step {int(times[frame])}')

        # Update marker position
        marker.set_data([times[frame]], [observable[frame]])
        vline.set_xdata([times[frame], times[frame]])

        return [im, spin_title, marker, vline]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(configurations),
        interval=interval,
        blit=True,
    )

    _save_animation(anim, filename, fps, dpi)

    plt.close(fig)


def create_temperature_sweep_animation(
    temperatures: np.ndarray,
    configurations: List[np.ndarray],
    filename: str,
    fps: int = 5,
    cmap: str = 'RdBu',
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 100,
    Tc: Optional[float] = None,
    interval: Optional[int] = None,
) -> None:
    """Create animation showing configurations across temperatures.

    Useful for visualizing the phase transition from ordered (low T)
    through critical (T ~ Tc) to disordered (high T).

    Parameters
    ----------
    temperatures : np.ndarray
        Temperature values for each configuration.
    configurations : list of np.ndarray
        Spin configurations at each temperature.
    filename : str
        Output file path.
    fps : int, optional
        Frames per second. Default is 5.
    cmap : str, optional
        Colormap. Default is 'RdBu'.
    figsize : tuple, optional
        Figure size. Default is (6, 6).
    dpi : int, optional
        Resolution. Default is 100.
    Tc : float, optional
        Critical temperature to highlight in title.
    interval : int, optional
        Delay between frames in milliseconds.

    Examples
    --------
    >>> temps = np.linspace(1.5, 3.5, 20)
    >>> configs = []
    >>> for T in temps:
    ...     model.temperature = T
    ...     sampler.run(1000)  # Equilibrate
    ...     configs.append(model.spins.copy())
    >>> create_temperature_sweep_animation(temps, configs, 'sweep.gif')
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        raise ImportError("matplotlib required for animation")

    if len(configurations) == 0:
        raise ValueError("At least one configuration required")

    if len(temperatures) != len(configurations):
        raise ValueError(
            f"temperatures ({len(temperatures)}) and configurations "
            f"({len(configurations)}) must have same length"
        )

    for i, config in enumerate(configurations):
        if config.ndim != 2:
            raise ValueError(
                f"Configuration {i} must be 2D, got {config.ndim}D"
            )

    if interval is None:
        interval = 1000 // fps

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        configurations[0],
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin='lower',
        interpolation='nearest',
    )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    # Create title with temperature info
    if Tc is not None:
        title_template = 'T = {T:.3f} (Tc = {Tc:.3f})'
        title = ax.set_title(title_template.format(T=temperatures[0], Tc=Tc))
    else:
        title_template = 'T = {T:.3f}'
        title = ax.set_title(title_template.format(T=temperatures[0]))

    def update(frame):
        im.set_array(configurations[frame])
        if Tc is not None:
            title.set_text(title_template.format(T=temperatures[frame], Tc=Tc))
        else:
            title.set_text(title_template.format(T=temperatures[frame]))
        return [im, title]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(configurations),
        interval=interval,
        blit=True,
    )

    _save_animation(anim, filename, fps, dpi)

    plt.close(fig)


def create_domain_growth_animation(
    configurations: List[np.ndarray],
    filename: str,
    times: Optional[np.ndarray] = None,
    fps: int = 10,
    cmap: str = 'RdBu',
    figsize: Tuple[float, float] = (10, 5),
    dpi: int = 100,
    interval: Optional[int] = None,
) -> None:
    """Create animation showing domain growth after quench.

    Shows spin configuration and domain size evolution side by side.
    Domain size is estimated from the correlation length.

    Parameters
    ----------
    configurations : list of np.ndarray
        Spin configurations during domain growth.
    filename : str
        Output file path.
    times : np.ndarray, optional
        Time values for each configuration. Default is frame indices.
    fps : int, optional
        Frames per second. Default is 10.
    cmap : str, optional
        Colormap. Default is 'RdBu'.
    figsize : tuple, optional
        Figure size. Default is (10, 5).
    dpi : int, optional
        Resolution. Default is 100.
    interval : int, optional
        Delay between frames in milliseconds.

    Examples
    --------
    >>> # Quench from high T to low T
    >>> model.randomize()
    >>> model.temperature = 1.5  # Below Tc
    >>> configs = [model.spins.copy()]
    >>> for i in range(1000):
    ...     sampler.step()
    ...     if i % 10 == 0:
    ...         configs.append(model.spins.copy())
    >>> create_domain_growth_animation(configs, 'quench.gif')
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        raise ImportError("matplotlib required for animation")

    if len(configurations) == 0:
        raise ValueError("At least one configuration required")

    for i, config in enumerate(configurations):
        if config.ndim != 2:
            raise ValueError(
                f"Configuration {i} must be 2D, got {config.ndim}D"
            )

    if times is None:
        times = np.arange(len(configurations))

    if len(times) != len(configurations):
        raise ValueError(
            f"times ({len(times)}) and configurations "
            f"({len(configurations)}) must have same length"
        )

    if interval is None:
        interval = 1000 // fps

    # Compute domain sizes (average absolute magnetization as proxy)
    domain_sizes = np.array([np.abs(np.mean(c)) for c in configurations])

    # Create figure
    fig, (ax_spin, ax_domain) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: spin configuration
    im = ax_spin.imshow(
        configurations[0],
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin='lower',
        interpolation='nearest',
    )
    ax_spin.set_xlabel('x')
    ax_spin.set_ylabel('y')
    ax_spin.set_aspect('equal')
    spin_title = ax_spin.set_title(f't = {times[0]:.0f}')

    # Right panel: domain size evolution
    ax_domain.plot(times, domain_sizes, 'b-', alpha=0.5, linewidth=1)
    marker, = ax_domain.plot(
        [times[0]], [domain_sizes[0]],
        'ro', markersize=8, zorder=10
    )
    ax_domain.set_xlabel('Time')
    ax_domain.set_ylabel('|Magnetization|')
    ax_domain.set_title('Domain Growth')
    ax_domain.grid(True, alpha=0.3)
    ax_domain.set_xlim(times[0], times[-1])
    ax_domain.set_ylim(0, 1.1)

    plt.tight_layout()

    def update(frame):
        im.set_array(configurations[frame])
        spin_title.set_text(f't = {times[frame]:.0f}')
        marker.set_data([times[frame]], [domain_sizes[frame]])
        return [im, spin_title, marker]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(configurations),
        interval=interval,
        blit=True,
    )

    _save_animation(anim, filename, fps, dpi)

    plt.close(fig)


def _save_animation(anim, filename: str, fps: int, dpi: int) -> None:
    """Save animation to file.

    Parameters
    ----------
    anim : FuncAnimation
        Animation object.
    filename : str
        Output file path.
    fps : int
        Frames per second.
    dpi : int
        Resolution.
    """
    import matplotlib.pyplot as plt

    filename = str(filename)
    ext = filename.lower().split('.')[-1]

    if ext == 'gif':
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            anim.save(filename, writer=writer, dpi=dpi)
        except ImportError:
            raise ImportError(
                "Pillow required for GIF output. Install with: pip install Pillow"
            )
    elif ext == 'mp4':
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(filename, writer=writer, dpi=dpi)
        except (ImportError, RuntimeError) as e:
            raise ImportError(
                f"FFmpeg required for MP4 output. Error: {e}"
            )
    else:
        # Try default writer
        anim.save(filename, fps=fps, dpi=dpi)


def _check_animation_dependencies() -> dict:
    """Check which animation writers are available.

    Returns
    -------
    dict
        Dictionary with 'gif' and 'mp4' keys indicating availability.
    """
    available = {'gif': False, 'mp4': False}

    try:
        from matplotlib.animation import PillowWriter
        available['gif'] = True
    except ImportError:
        pass

    try:
        from matplotlib.animation import FFMpegWriter
        # Check if ffmpeg is actually available
        import subprocess
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        available['mp4'] = result.returncode == 0
    except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return available
