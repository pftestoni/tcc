import argparse
import csv
from pathlib import Path
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO
from csp_rl.envs.ctl_env import make_env_from_yaml


def evaluate(env, model, episodes: int = 20, seed: int = 123):
    rng = np.random.default_rng(seed)
    metrics = {
        "retalho_pct": [],
        "estoque_medio": [],
        "fill_rate": [],
        "setups_semana": [],
        "tempo_inferencia_ms": [],
        "coils_por_semana": []
    }
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        ep = {
            "waste_total": 0.0,
            "stock_acc": 0.0,
            "steps": 0,
            "sold": 0,
            "demand": 0,
            "produced_m": 0.0,
            "setups": 0,
            "coils": 0
        }
        import time as _t
        t0 = _t.time()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep["waste_total"] = info["waste_m_total"]
            ep["stock_acc"] += info["stock_units"]
            ep["steps"] += 1
            ep["sold"] = env.total_sold
            ep["demand"] = env.total_demand
            ep["produced_m"] = env.total_produced_m
            ep["setups"] = env.setups_done
            ep["coils"] = info["coils_used"]
            if terminated or truncated:
                break
        infer_ms = ( _t.time() - t0 ) * 1000.0 / max(1, ep["steps"])

        ret_pct = 0.0 if ep["produced_m"] <= 1e-9 else 100.0 * ep["waste_total"] / ep["produced_m"]
        estoque_med = ep["stock_acc"] / max(1, ep["steps"])
        fill = 0.0 if ep["demand"] <= 0 else ep["sold"] / ep["demand"]
        setups_sem = ep["setups"] / env.horizon_weeks
        coils_sem = ep["coils"] / env.horizon_weeks

        metrics["retalho_pct"].append(ret_pct)
        metrics["estoque_medio"].append(estoque_med)
        metrics["fill_rate"].append(fill)
        metrics["setups_semana"].append(setups_sem)
        metrics["tempo_inferencia_ms"].append(infer_ms)
        metrics["coils_por_semana"].append(coils_sem)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def find_latest_model(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    # pega subpasta mais recente (ex.: ppo_YYYYMMDD_HHMMSS) que contenha model.zip
    subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for d in subdirs:
        m = d / "model.zip"
        if m.exists():
            return m
    return None


def main():
    parser = argparse.ArgumentParser(description="Avaliar modelo PPO salvo para CTL 1D.")
    parser.add_argument("--model", type=str, default=None, help="Caminho para model.zip. Se omitido, usa o último em reports/runs/.")
    parser.add_argument("--yaml", type=str, default="src/csp_rl/configs/default.yaml", help="Config YAML do ambiente.")
    parser.add_argument("--episodes", type=int, default=50, help="Nº de episódios de avaliação.")
    parser.add_argument("--seed", type=int, default=999, help="Seed base da avaliação.")
    parser.add_argument("--out", type=str, default=None, help="Caminho para salvar CSV (opcional). Padrão: ao lado do model.zip.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    runs_dir = root / "reports" / "runs"

    model_path = Path(args.model) if args.model else find_latest_model(runs_dir)
    if model_path is None or not model_path.exists():
        raise SystemExit("Nenhum model.zip encontrado. Treine primeiro ou passe --model.")

    env = make_env_from_yaml(args.yaml, seed=123)
    model = PPO.load(model_path)

    results = evaluate(env, model, episodes=args.episodes, seed=args.seed)

    # Print bonito
    print("===== Avaliação (médias) =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print(f"\nmodel: {model_path}")

    # Salvar CSV (local padrão: mesma pasta do modelo)
    out_csv = Path(args.out) if args.out else (model_path.parent / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    out_csv = out_csv.with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in results.items():
            w.writerow([k, f"{v:.6f}"])
    print(f"CSV salvo em: {out_csv}")

if __name__ == "__main__":
    main()
