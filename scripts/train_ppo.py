import os
from datetime import datetime
from pathlib import Path
import csv
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from csp_rl.envs.ctl_env import make_env_from_yaml


def evaluate(env, model, episodes: int = 20, seed: int = 2025):
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


def main():
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "src" / "csp_rl" / "configs" / "default.yaml"
    runs_dir = root / "reports" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Ambiente
    def _factory():
        return make_env_from_yaml(cfg_path, seed=2025)
    vec_env = DummyVecEnv([_factory])

    # Treino PPO
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=2025)
    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps)

    # Salvar
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = runs_dir / f"ppo_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.zip")

    # Avaliação rápida
    # Reinstancia env single (não vetorizado) para avaliar
    env_eval = make_env_from_yaml(cfg_path, seed=123)
    results = evaluate(env_eval, model, episodes=20, seed=999)

    # Salvar métricas em CSV
    csv_path = out_dir / "eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in results.items():
            w.writerow([k, f"{v:.6f}"])

    print("===== Avaliação (médias em 20 episódios) =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print(f"\nArquivos salvos em: {out_dir}")

if __name__ == "__main__":
    main()
