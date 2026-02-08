"""
Create animated GIF of spin evolution during Monte Carlo simulation.

Reads spin configurations from HDF5 file and generates animation.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap


def create_spin_animation(spin_history, output_file, fps=10, duration=5):
    """Create animated GIF from spin configuration history.

    Parameters
    ----------
    spin_history : np.ndarray
        Array of shape (n_frames, L, L) with spin configurations
    output_file : str
        Output GIF file path
    fps : int
        Frames per second
    duration : float
        Total animation duration in seconds
    """
    n_frames = len(spin_history)
    L = spin_history[0].shape[0]

    # Calculate frame interval
    total_frames = int(fps * duration)
    if n_frames > total_frames:
        # Sample frames evenly
        indices = np.linspace(0, n_frames - 1, total_frames, dtype=int)
        spin_history = spin_history[indices]
        n_frames = total_frames

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Custom colormap: blue for spin down, red for spin up
    cmap = ListedColormap(['#3498db', '#e74c3c'])

    # Initialize plot
    im = ax.imshow(spin_history[0], cmap=cmap, vmin=-1, vmax=1,
                   interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title(f'Frame 0/{n_frames}', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 1], shrink=0.8)
    cbar.ax.set_yticklabels([r'$\downarrow$', r'$\uparrow$'], fontsize=14)

    def update(frame):
        """Update function for animation."""
        im.set_array(spin_history[frame])
        title.set_text(f'Frame {frame + 1}/{n_frames}')
        return [im, title]

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 / fps, blit=True
    )

    # Save as GIF
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anim.save(output_file, writer='pillow', fps=fps)
    plt.close()

    print(f"Animation saved to {output_file}")
    print(f"  Frames: {n_frames}, FPS: {fps}, Duration: {n_frames/fps:.1f}s")


def main():
    """Create animation from simulation data."""
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    fps = snakemake.params.get('fps', 10)
    duration = snakemake.params.get('duration', 5)

    # Load spin history from HDF5
    with h5py.File(input_file, 'r') as f:
        if 'spin_history' in f:
            spin_history = f['spin_history'][:]
        elif 'spins' in f:
            # Single configuration - create simple animation
            spins = f['spins'][:]
            spin_history = np.array([spins])
        else:
            raise ValueError("No spin data found in HDF5 file")

        # Get metadata
        if 'temperature' in f.attrs:
            temperature = f.attrs['temperature']
        else:
            temperature = None

        if 'size' in f.attrs:
            size = f.attrs['size']
        else:
            size = spin_history[0].shape[0]

    print(f"Loaded {len(spin_history)} frames, L={size}")
    if temperature is not None:
        print(f"Temperature: {temperature}")

    # Create animation
    create_spin_animation(spin_history, output_file, fps=fps, duration=duration)


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1]]
            self.output = [sys.argv[2]]
            self.params = {'fps': 10, 'duration': 5}

    snakemake = MockSnakemake()
    main()
