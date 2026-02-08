"""
Generate HTML summary report from simulation results.
"""

import base64
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def embed_image(filepath):
    """Embed image as base64 in HTML."""
    filepath = Path(filepath)
    if not filepath.exists():
        return ""

    with open(filepath, 'rb') as f:
        data = base64.b64encode(f.read()).decode()

    suffix = filepath.suffix.lower()
    if suffix == '.pdf':
        # PDFs can't be embedded directly, use object tag
        return f'<object data="data:application/pdf;base64,{data}" type="application/pdf" width="100%" height="400px"></object>'
    else:
        mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'svg': 'image/svg+xml'}.get(suffix[1:], 'image/png')
        return f'<img src="data:{mime};base64,{data}" style="max-width:100%;">'


def main():
    """Generate HTML report."""
    obs_file = snakemake.input.observables
    critical_file = snakemake.input.critical
    phase_diagram = snakemake.input.phase_diagram
    binder_plot = snakemake.input.binder
    fss_plot = snakemake.input.fss
    output_file = snakemake.output[0]

    # Load data
    df = pd.read_csv(obs_file)
    with open(critical_file) as f:
        critical = json.load(f)

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ising Monte Carlo Simulation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f2f2f2;
        }}
        .figure {{
            text-align: center;
            margin: 20px 0;
        }}
        .highlight {{
            background: #e7f3ff;
            padding: 10px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>Ising Monte Carlo Simulation Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="card">
        <h2>Simulation Parameters</h2>
        <ul>
            <li><strong>Model:</strong> {critical.get('model', 'ising2d')}</li>
            <li><strong>Sizes:</strong> {critical.get('sizes', [])}</li>
            <li><strong>Temperature range:</strong> {critical.get('temperature_range', [])}</li>
            <li><strong>Number of temperatures:</strong> {critical.get('n_temperatures', 0)}</li>
        </ul>
    </div>

    <div class="card">
        <h2>Critical Point Analysis</h2>
        <div class="highlight">
            <h3>Estimated Critical Temperature</h3>
            <p style="font-size: 24px; text-align: center;">
                T<sub>c</sub> = {critical.get('Tc_estimate', 0):.4f} ± {critical.get('binder_analysis', {}).get('Tc_error', 0):.4f}
            </p>
            <p style="text-align: center;">
                (Exact value for 2D Ising: T<sub>c</sub> = 2.2692)
            </p>
        </div>

        <h3>Susceptibility Peaks</h3>
        <table>
            <tr><th>Size (L)</th><th>T<sub>peak</sub></th><th>χ<sub>max</sub></th></tr>
"""

    for size, data in sorted(critical.get('susceptibility_peaks', {}).items()):
        html += f"<tr><td>{size}</td><td>{data['T_peak']:.4f}</td><td>{data['chi_peak']:.4f}</td></tr>\n"

    html += """
        </table>

        <h3>Binder Crossings</h3>
        <table>
            <tr><th>L<sub>1</sub></th><th>L<sub>2</sub></th><th>T<sub>cross</sub></th><th>U<sub>cross</sub></th></tr>
"""

    for crossing in critical.get('binder_analysis', {}).get('crossings', []):
        html += f"<tr><td>{crossing['L1']}</td><td>{crossing['L2']}</td>"
        html += f"<td>{crossing['T_cross']:.4f}</td><td>{crossing['U_cross']:.4f}</td></tr>\n"

    html += f"""
        </table>
    </div>

    <div class="card">
        <h2>Phase Diagram</h2>
        <div class="figure">
            {embed_image(phase_diagram)}
        </div>
    </div>

    <div class="card">
        <h2>Binder Cumulant Crossing</h2>
        <div class="figure">
            {embed_image(binder_plot)}
        </div>
    </div>

    <div class="card">
        <h2>Finite-Size Scaling</h2>
        <div class="figure">
            {embed_image(fss_plot)}
        </div>
    </div>

    <div class="card">
        <h2>Data Summary</h2>
        <p>Total simulations: {len(df)}</p>
        {df.to_html(index=False, max_rows=20)}
    </div>

</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Report saved to {output_file}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        class Input:
            def __init__(self, args):
                self.observables = args[0]
                self.critical = args[1]
                self.phase_diagram = args[2]
                self.binder = args[3]
                self.fss = args[4]

        def __init__(self):
            self.input = self.Input(sys.argv[2:7])
            self.output = [sys.argv[1]]

    snakemake = MockSnakemake()
    main()
