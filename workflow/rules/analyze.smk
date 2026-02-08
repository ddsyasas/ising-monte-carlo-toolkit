"""
Analysis rules for Ising Monte Carlo workflow.

This file contains rules for:
- Calculating observables with error estimates
- Aggregating sweep results
- Finding critical temperature
- Extracting critical exponents
"""

# =============================================================================
# Observable calculation
# =============================================================================

rule calculate_observables:
    """Calculate observables with bootstrap error estimates for a single simulation."""
    input:
        f"{RESULTS_DIR}/{{model}}_L{{size}}_T{{temp}}.npz"
    output:
        f"{RESULTS_DIR}/analysis/{{model}}_L{{size}}_T{{temp}}_obs.csv"
    params:
        bootstrap=config.get("bootstrap_samples", 1000)
    log:
        f"logs/obs_{{model}}_L{{size}}_T{{temp}}.log"
    script:
        "../scripts/calculate_observables.py"


rule aggregate_observables:
    """Aggregate observable data across temperatures for a single size."""
    input:
        expand(f"{RESULTS_DIR}/analysis/{{{{model}}}}_L{{{{size}}}}_T{{temp}}_obs.csv",
               temp=TEMPS)
    output:
        f"{RESULTS_DIR}/{{model}}_L{{size}}_sweep.csv"
    log:
        f"logs/aggregate_{{model}}_L{{size}}.log"
    script:
        "../scripts/aggregate_sweep.py"


# =============================================================================
# Critical point analysis
# =============================================================================

rule find_critical_temperature:
    """Estimate critical temperature from multiple system sizes."""
    input:
        expand(f"{RESULTS_DIR}/{{{{model}}}}_L{{size}}_sweep.csv", size=SIZES)
    output:
        f"{RESULTS_DIR}/{{model}}_critical_temp.json"
    params:
        method=config.get("tc_method", "binder_crossing"),
        sizes=SIZES
    log:
        f"logs/find_Tc_{{model}}.log"
    script:
        "../scripts/find_Tc.py"


rule extract_exponents:
    """Extract critical exponents from finite-size scaling analysis."""
    input:
        sweeps=expand(f"{RESULTS_DIR}/{{{{model}}}}_L{{size}}_sweep.csv", size=SIZES),
        Tc=f"{RESULTS_DIR}/{{model}}_critical_temp.json"
    output:
        f"{RESULTS_DIR}/{{model}}_exponents.json"
    params:
        model="{model}",
        sizes=SIZES
    log:
        f"logs/exponents_{{model}}.log"
    script:
        "../scripts/extract_exponents.py"


# =============================================================================
# Derived quantities
# =============================================================================

rule compute_binder_cumulant:
    """Compute Binder cumulant U = 1 - <m^4>/(3<m^2>^2) for each simulation."""
    input:
        f"{RESULTS_DIR}/{{model}}_L{{size}}_T{{temp}}.npz"
    output:
        f"{RESULTS_DIR}/analysis/{{model}}_L{{size}}_T{{temp}}_binder.csv"
    params:
        bootstrap=config.get("bootstrap_samples", 1000)
    script:
        "../scripts/compute_binder_single.py"


rule aggregate_binder:
    """Aggregate Binder cumulant data for finite-size scaling."""
    input:
        expand(f"{RESULTS_DIR}/analysis/{{{{model}}}}_L{{size}}_T{{temp}}_binder.csv",
               size=SIZES, temp=TEMPS)
    output:
        f"{RESULTS_DIR}/{{model}}_binder_all.csv"
    script:
        "../scripts/aggregate_binder.py"


rule compute_correlation_length:
    """Estimate correlation length from spin-spin correlations."""
    input:
        f"{RESULTS_DIR}/{{model}}_L{{size}}_T{{temp}}.npz"
    output:
        f"{RESULTS_DIR}/analysis/{{model}}_L{{size}}_T{{temp}}_xi.csv"
    params:
        method=config.get("xi_method", "second_moment")
    script:
        "../scripts/compute_xi.py"


# =============================================================================
# Analysis targets
# =============================================================================

rule all_observables:
    """Calculate all observables for all simulations."""
    input:
        expand(f"{RESULTS_DIR}/analysis/{{model}}_L{{size}}_T{{temp}}_obs.csv",
               model=MODEL, size=SIZES, temp=TEMPS)


rule all_sweeps:
    """Generate sweep files for all sizes."""
    input:
        expand(f"{RESULTS_DIR}/{{model}}_L{{size}}_sweep.csv",
               model=MODEL, size=SIZES)


rule critical_analysis:
    """Complete critical analysis including Tc and exponents."""
    input:
        Tc=expand(f"{RESULTS_DIR}/{{model}}_critical_temp.json", model=MODEL),
        exponents=expand(f"{RESULTS_DIR}/{{model}}_exponents.json", model=MODEL),
        binder=expand(f"{RESULTS_DIR}/{{model}}_binder_all.csv", model=MODEL)
