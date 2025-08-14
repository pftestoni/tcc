import argparse
from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt

def find_eval_csv(run_dir: Path) -> Path | None:
    # Prioriza eval.csv; se não existir, pega o eval_*.csv mais recente
    p1 = run_dir / "eval.csv"
    if p1.exists():
        return p1
    candidates = sorted(run_dir.glob("eval_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None

def load_runs(runs_dir: Path) -> pd.DataFrame:
    if not runs_dir.exists():
        raise SystemExit(f"Nenhum diretório: {runs_dir}")
    rows = []
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise SystemExit("Nenhum run encontrado em reports/runs/")
    # Ordena por mtime crescente (histórico)
    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    for i, rd in enumerate(run_dirs, start=1):
        csv_path = find_eval_csv(rd)
        if not csv_path:
            continue
        try:
            df = pd.read_csv(csv_path)
            if {"metric","value"}.issubset(df.columns):
                rec = {r["metric"]: float(str(r["value"]).replace(",",".")) for _, r in df.iterrows()}
            else:
                # fallback: duas colunas quaisquer
                rec = {row[0]: float(str(row[1]).replace(",",".")) for row in df.itertuples(index=False)}
        except Exception:
            continue
        rec["run_name"] = rd.name
        rec["run_path"] = str(rd)
        rec["mtime"] = datetime.fromtimestamp(rd.stat().st_mtime)
        rec["idx"] = i
        rows.append(rec)
    if not rows:
        raise SystemExit("Nenhum eval.csv encontrado nos runs.")
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)

def plot_metrics(df: pd.DataFrame, out_dir: Path, metrics: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = df["idx"]
    labels = df["run_name"]
    for m in metrics:
        if m not in df.columns:
            continue
        plt.figure(figsize=(10,5), dpi=150)
        plt.plot(x, df[m], marker="o")
        plt.title(m)
        plt.xlabel("run (ordem cronológica)")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.tight_layout()
        out_path = out_dir / f"{m}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"salvo: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plotar métricas dos runs (eval.csv) em reports/runs/")
    parser.add_argument("--runs", default="reports/runs", help="Diretório com runs")
    parser.add_argument("--out",  default="reports/figs", help="Diretório de saída para figuras")
    parser.add_argument("--metrics", nargs="*", default=["retalho_pct","estoque_medio","fill_rate","setups_semana","coils_por_semana"],
                        help="Lista de métricas a plotar")
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    out_dir = Path(args.out)

    df = load_runs(runs_dir)
    # salva um resumo em CSV também
    summary_csv = out_dir / "summary_runs.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    print(f"resumo salvo: {summary_csv}")

    plot_metrics(df, out_dir, args.metrics)
    print("ok.")

if __name__ == "__main__":
    main()
