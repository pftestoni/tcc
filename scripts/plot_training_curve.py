import argparse
from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt

def find_latest_run(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    runs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]

def load_train_history(run_dir: Path) -> pd.DataFrame:
    # prefere train_history.csv; se não houver, tenta *.monitor.csv
    hist = run_dir / "train_history.csv"
    if hist.exists():
        df = pd.read_csv(hist)
        return df
    # fallback: monitor
    monitors = sorted(run_dir.glob("*.monitor.csv"))
    if not monitors:
        raise SystemExit("Nenhum histórico encontrado (train_history.csv nem *.monitor.csv).")
    # concatena todos os monitores
    frames = []
    for m in monitors:
        rows = []
        with m.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    try:
                        r = float(parts[0]); l = int(float(parts[1])); t = float(parts[2])
                        rows.append((r,l,t))
                    except ValueError:
                        pass
        if rows:
            dfm = pd.DataFrame(rows, columns=["ep_reward","ep_length","t_start"])
            frames.append(dfm)
    if not frames:
        raise SystemExit("Monitor encontrado, mas vazio.")
    df = pd.concat(frames, ignore_index=True)
    df["episode"] = range(1, len(df)+1)
    return df

def smooth(series, k=10):
    if k <= 1:
        return series
    return series.rolling(window=k, min_periods=1).mean()

def main():
    parser = argparse.ArgumentParser(description="Plotar curva de evolução (recompensa por episódio) de um run.")
    parser.add_argument("--run", type=str, default=None, help="Pasta do run (ex.: reports/runs/ppo_YYYYMMDD_HHMMSS). Se omitido, usa o último.")
    parser.add_argument("--smooth", type=int, default=10, help="Janela de suavização (rolling mean).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    runs_dir = root / "reports" / "runs"

    run_dir = Path(args.run) if args.run else find_latest_run(runs_dir)
    if run_dir is None or not run_dir.exists():
        raise SystemExit("Nenhum run encontrado.")
    print(f"Run selecionado: {run_dir}")

    df = load_train_history(run_dir)
    df["ep_reward_smooth"] = smooth(df["ep_reward"], k=args.smooth)

    out_dir = root / "reports" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"train_curve_{run_dir.name}.png"

    plt.figure(figsize=(10,5), dpi=150)
    plt.plot(df["episode"], df["ep_reward"], label="Reward por episódio")
    plt.plot(df["episode"], df["ep_reward_smooth"], label=f"Suavizado (k={args.smooth})")
    plt.title(f"Evolução do treino — {run_dir.name}")
    plt.xlabel("Episódio (treino)")
    plt.ylabel("Recompensa do episódio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Figura salva em: {out_png}")

if __name__ == "__main__":
    main()
