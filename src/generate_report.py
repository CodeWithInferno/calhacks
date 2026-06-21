"""Generate an interactive HTML training report from training_log.csv."""

import json
import argparse
import pandas as pd


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>G1 World Model Training Report</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2rem; background: #f8f9fa; }}
    h1 {{ color: #212529; }}
    .metric {{ display: inline-block; margin: 0.5rem 1rem 0.5rem 0; padding: 0.8rem 1.2rem; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .metric .value {{ font-size: 1.6rem; font-weight: bold; color: #0d6efd; }}
    .metric .label {{ font-size: 0.85rem; color: #6c757d; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }}
    .card {{ background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  </style>
</head>
<body>
  <h1>G1 Load & Slope Risk Copilot — Training Report</h1>
  <div>
    <div class="metric"><div class="value">{best_auc:.4f}</div><div class="label">Best Val AUC</div></div>
    <div class="metric"><div class="value">{final_auc:.4f}</div><div class="label">Final Val AUC</div></div>
    <div class="metric"><div class="value">{final_f1:.4f}</div><div class="label">Final Val F1</div></div>
    <div class="metric"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
    <div class="metric"><div class="value">{total_time:.1f}s</div><div class="label">Total Time</div></div>
  </div>
  <div class="grid">
    <div class="card" id="loss"></div>
    <div class="card" id="ranking"></div>
    <div class="card" id="classification"></div>
    <div class="card" id="lr"></div>
  </div>
  <script>
    const data = {json_data};

    const trace = (name, y, color, dash='solid') => ({{
      x: data.epoch,
      y: y,
      mode: 'lines',
      name: name,
      line: {{ color: color, width: 2, dash: dash }}
    }});

    Plotly.newPlot('loss', [
      trace('train loss', data.train_loss, '#0d6efd'),
      trace('val loss', data.val_loss, '#dc3545')
    ], {{
      title: 'Loss',
      xaxis: {{ title: 'epoch' }},
      yaxis: {{ title: 'loss' }},
      hovermode: 'x unified'
    }}, {{responsive: true}});

    Plotly.newPlot('ranking', [
      trace('val AUC', data.val_auc, '#198754'),
      trace('val AP', data.val_ap, '#fd7e14')
    ], {{
      title: 'Validation Ranking Metrics',
      xaxis: {{ title: 'epoch' }},
      yaxis: {{ title: 'score' }},
      hovermode: 'x unified'
    }}, {{responsive: true}});

    Plotly.newPlot('classification', [
      trace('val accuracy', data.val_acc, '#0dcaf0'),
      trace('val F1', data.val_f1, '#6f42c1')
    ], {{
      title: 'Validation Classification Metrics',
      xaxis: {{ title: 'epoch' }},
      yaxis: {{ title: 'score' }},
      hovermode: 'x unified'
    }}, {{responsive: true}});

    Plotly.newPlot('lr', [
      trace('learning rate', data.lr, '#6610f2')
    ], {{
      title: 'Learning Rate',
      xaxis: {{ title: 'epoch' }},
      yaxis: {{ title: 'lr' }},
      hovermode: 'x unified'
    }}, {{responsive: true}});
  </script>
</body>
</html>
"""


def generate(log_path, output_path):
    df = pd.read_csv(log_path)
    data = {col: df[col].tolist() for col in df.columns}

    best_auc = df["val_auc"].max()
    final_auc = df["val_auc"].iloc[-1]
    final_f1 = df["val_f1"].iloc[-1]
    epochs = int(df["epoch"].iloc[-1])
    total_time = df["epoch_time"].astype(float).sum()

    html = HTML_TEMPLATE.format(
        json_data=json.dumps(data),
        best_auc=best_auc,
        final_auc=final_auc,
        final_f1=final_f1,
        epochs=epochs,
        total_time=total_time,
    )

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="results/training_log.csv")
    parser.add_argument("--output", default="results/training_report.html")
    args = parser.parse_args()
    generate(args.log, args.output)


if __name__ == "__main__":
    main()
