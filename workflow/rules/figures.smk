"""
Visualization rules for Ising Monte Carlo workflow.

This file contains rules for:
- Phase diagram plots
- Binder cumulant crossing analysis
- Finite-size scaling collapse
- Spin configuration animations
"""

# =============================================================================
# Static figures
# =============================================================================

rule plot_phase_diagram:
    """Generate phase diagram showing M(T) and chi(T) for all sizes."""
    input:
        expand(f"{RESULTS_DIR}/{{{{model}}}}_L{{size}}_sweep.csv", size=SIZES)
    output:
        f"{FIGURES_DIR}/{{model}}_phase_diagram.pdf"
    params:
        sizes=SIZES,
        style=config.get("plot_style", "publication")
    log:
        f"logs/plot_phase_{{model}}.log"
    script:
        "../scripts/plot_phase_diagram.py"


rule plot_binder_crossing:
    """Plot Binder cumulant U(T) for multiple sizes to find crossing point."""
    input:
        expand(f"{RESULTS_DIR}/{{{{model}}}}_L{{size}}_sweep.csv", size=SIZES)
    output:
        f"{FIGURES_DIR}/{{model}}_binder_crossing.pdf"
    params:
        sizes=SIZES,
        style=config.get("plot_style", "publication")
    log:
        f"logs/plot_binder_{{model}}.log"
    script:
        "../scripts/plot_binder.py"


rule plot_scaling_collapse:
    """Generate finite-size scaling collapse plot."""
    input:
        sweeps=expand(f"{RESULTS_DIR}/{{{{model}}}}_L{{size}}_sweep.csv", size=SIZES),
        exponents=f"{RESULTS_DIR}/{{model}}_exponents.json"
    output:
        f"{FIGURES_DIR}/{{model}}_scaling_collapse.pdf"
    params:
        sizes=SIZES,
        style=config.get("plot_style", "publication")
    log:
        f"logs/plot_collapse_{{model}}.log"
    script:
        "../scripts/plot_collapse.py"


# =============================================================================
# Animations
# =============================================================================

rule create_animation:
    """Create animated GIF of spin evolution during simulation."""
    input:
        f"{RESULTS_DIR}/{{model}}_L{{size}}_T{{temp}}.h5"
    output:
        f"{FIGURES_DIR}/animations/{{model}}_L{{size}}_T{{temp}}.gif"
    params:
        fps=config.get("animation_fps", 10),
        duration=config.get("animation_duration", 5)
    log:
        f"logs/animation_{{model}}_L{{size}}_T{{temp}}.log"
    script:
        "../scripts/create_animation.py"


# =============================================================================
# Additional visualization rules
# =============================================================================

rule plot_susceptibility:
    """Plot magnetic susceptibility vs temperature."""
    input:
        expand(f"{RESULTS_DIR}/{{{{model}}}}_L{{size}}_sweep.csv", size=SIZES)
    output:
        f"{FIGURES_DIR}/{{model}}_susceptibility.pdf"
    params:
        sizes=SIZES,
        style=config.get("plot_style", "publication")
    script:
        "../scripts/plot_susceptibility.py"


rule plot_specific_heat:
    """Plot specific heat vs temperature."""
    input:
        expand(f"{RESULTS_DIR}/{{{{model}}}}_L{{size}}_sweep.csv", size=SIZES)
    output:
        f"{FIGURES_DIR}/{{model}}_specific_heat.pdf"
    params:
        sizes=SIZES,
        style=config.get("plot_style", "publication")
    script:
        "../scripts/plot_specific_heat.py"


rule plot_snapshots:
    """Generate grid of spin configuration snapshots at different temperatures."""
    input:
        expand(f"{RESULTS_DIR}/{{{{model}}}}_L{{size}}_T{{temp}}.npz",
               size=max(SIZES), temp=[TEMPS[0], TEMPS[len(TEMPS)//2], TEMPS[-1]])
    output:
        f"{FIGURES_DIR}/{{model}}_snapshots.pdf"
    params:
        style=config.get("plot_style", "publication")
    script:
        "../scripts/plot_snapshots.py"


rule plot_correlation_function:
    """Plot spin-spin correlation function C(r)."""
    input:
        f"{RESULTS_DIR}/{{model}}_L{{size}}_T{{temp}}.npz"
    output:
        f"{FIGURES_DIR}/correlation/{{model}}_L{{size}}_T{{temp}}_Cr.pdf"
    params:
        style=config.get("plot_style", "publication")
    script:
        "../scripts/plot_correlation.py"


# =============================================================================
# Figure targets
# =============================================================================

rule all_figures:
    """Generate all publication figures."""
    input:
        expand(f"{FIGURES_DIR}/{{model}}_phase_diagram.pdf", model=MODEL),
        expand(f"{FIGURES_DIR}/{{model}}_binder_crossing.pdf", model=MODEL),
        expand(f"{FIGURES_DIR}/{{model}}_scaling_collapse.pdf", model=MODEL),
        expand(f"{FIGURES_DIR}/{{model}}_susceptibility.pdf", model=MODEL),
        expand(f"{FIGURES_DIR}/{{model}}_specific_heat.pdf", model=MODEL),
        expand(f"{FIGURES_DIR}/{{model}}_snapshots.pdf", model=MODEL)


rule all_animations:
    """Generate all spin evolution animations."""
    input:
        expand(f"{FIGURES_DIR}/animations/{{model}}_L{{size}}_T{{temp}}.gif",
               model=MODEL, size=SIZES, temp=TEMPS)
