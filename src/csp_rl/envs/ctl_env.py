import math
import time
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml


# =========================
# Utilidades de configuração
# =========================
def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================
# Gerador de demanda
# =========================
def truncated_normal_int(mean: float, std: float, min_positive: int = 1) -> int:
    x = np.random.normal(loc=mean, scale=max(1e-9, std))
    # Trunca para garantir positividade mínima
    val = int(round(x))
    return max(min_positive, val)


class DemandGenerator:
    """
    Gera matriz de demanda semanal (chapas) por comprimento.
    """
    def __init__(
        self,
        lengths_mm: np.ndarray,
        means_per_item: np.ndarray,
        sigma_frac: float = 0.15,
        horizon_weeks: int = 12,
        min_positive: int = 1,
        seed: Optional[int] = None,
    ):
        self.lengths_mm = lengths_mm.astype(np.int32)
        self.n_items = len(lengths_mm)
        self.means = means_per_item.astype(float)
        self.sigma = np.maximum(1e-9, sigma_frac * self.means)
        self.horizon_weeks = int(horizon_weeks)
        self.min_positive = int(min_positive)
        self.rng = np.random.default_rng(seed)

    def sample_episode(self) -> np.ndarray:
        weeks = []
        for _ in range(self.horizon_weeks):
            week = [truncated_normal_int(self.means[i], self.sigma[i], self.min_positive)
                    for i in range(self.n_items)]
            weeks.append(week)
        return np.array(weeks, dtype=np.int32)


# =========================
# Ambiente Gym: CTL 1D
# =========================
class CtlCuttingEnv(gym.Env):
    """
    CTL (cut-to-length) 1D:
      - A cada passo, o agente escolhe um comprimento e um lote (em metros) para produzir.
      - A produção consome capacidade semanal (m) e também consome 'bobinas-mãe'.
      - Cada nova bobina tem comprimento fixo (coil_length_m) e gera sucata fixa = frac * coil_length_m.
      - O lote produzido puxa 'lot_m' da(s) bobina(s); chapas produzidas = floor(lot_m / length_m).
      - Retalhos (waste):
          * sucata fixa de bobina (3% por bobina aberta)
          * sobra de lote (< 1 chapa): lot_m - produced_sheets * length_m
      - Sem backlog: vendas perdidas.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        lengths_mm: List[int],
        lot_bins_m: List[int],
        weekly_minutes: int,
        speed_m_per_min: float,
        horizon_weeks: int,
        forecast_k: int,
        coil_length_m: float,
        coil_fixed_scrap_frac: float,
        demand_cfg: Dict[str, Any],
        reward_weights: Dict[str, float] | None = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # Itens/Comprimentos
        self.lengths_mm = np.array(lengths_mm, dtype=np.int32)
        self.lengths_m = self.lengths_mm / 1000.0
        self.n_items = int(self.lengths_mm.size)

        # Lotes discretos (em metros)
        self.lot_bins_m = np.array(lot_bins_m, dtype=np.int32)

        # Capacidade semanal
        self.weekly_minutes = int(weekly_minutes)
        self.speed = float(speed_m_per_min)
        self.weekly_capacity_m = self.weekly_minutes * self.speed

        # Episódio e previsão
        self.horizon_weeks = int(horizon_weeks)
        self.forecast_k = int(forecast_k)

        # Bobina-mãe
        self.coil_length_m_total = float(coil_length_m)
        self.coil_fixed_scrap_frac = float(coil_fixed_scrap_frac)
        self.coil_fixed_scrap_m = self.coil_length_m_total * self.coil_fixed_scrap_frac

        # Recompensa
        if reward_weights is None:
            reward_weights = {"waste": 1.0, "stock": 0.02, "setups": 0.1, "viol": 10.0}
        self.w_waste = float(reward_weights.get("waste", 1.0))
        self.w_stock = float(reward_weights.get("stock", 0.02))
        self.w_setups = float(reward_weights.get("setups", 0.1))
        self.w_viol = float(reward_weights.get("viol", 10.0))

        # Gerador de demanda
        self.dem_gen = DemandGenerator(
            self.lengths_mm,
            means_per_item=np.array(demand_cfg["means_per_item"], dtype=float),
            sigma_frac=float(demand_cfg.get("sigma_frac", 0.15)),
            horizon_weeks=self.horizon_weeks,
            min_positive=int(demand_cfg.get("min_positive", 1)),
            seed=seed,
        )

        # Espaços
        # Observação: [estoque (n), forecast k semanas (k*n), cap_norm, last_idx_norm, setups_norm]
        obs_dim = self.n_items + self.n_items * self.forecast_k + 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.n_items, len(self.lot_bins_m)])

        # Estado
        self.reset_state()

    # -----------------
    # Ciclo de episódio
    # -----------------
    def reset_state(self):
        self.week_idx = 0
        self.capacity_left_m = self.weekly_capacity_m
        self.inventory = np.zeros(self.n_items, dtype=np.int32)
        self.last_idx = -1
        self.setups_done = 0

        # Abre a primeira bobina já debitando a sucata fixa
        self.coils_used = 0
        self.coil_remaining_m = 0.0
        self.total_waste_m = 0.0
        self._open_new_coil()

        # Demanda do episódio
        self.episode_dem = self.dem_gen.sample_episode()
        self.total_demand = 0
        self.total_sold = 0
        self.total_produced_m = 0.0
        self.total_steps = 0

        self._update_weekly_views()

    def _open_new_coil(self):
        # Aplica sucata fixa de 3% e inicia saldo da bobina
        self.coils_used += 1
        self.total_waste_m += self.coil_fixed_scrap_m
        self.coil_remaining_m = max(0.0, self.coil_length_m_total - self.coil_fixed_scrap_m)

    def _update_weekly_views(self):
        self.current_dem = self.episode_dem[self.week_idx].copy()
        self.total_demand += int(self.current_dem.sum())

        fut = []
        for i in range(1, self.forecast_k + 1):
            if self.week_idx + i < self.horizon_weeks:
                fut.append(self.episode_dem[self.week_idx + i])
            else:
                fut.append(np.zeros_like(self.current_dem))
        self.future_forecast = np.stack(fut, axis=0) if len(fut) > 0 else np.zeros((0, self.n_items), dtype=np.int32)

    def _obs(self) -> np.ndarray:
        # Normalizações simples
        dem_safe = np.maximum(1, self.current_dem)
        inv_norm = np.minimum(1.0, self.inventory / dem_safe)

        if self.future_forecast.size > 0:
            # Normaliza cada entrada do forecast pela demanda corrente do respectivo item
            fut = self.future_forecast / dem_safe[np.newaxis, :]
            fut_norm = np.clip(fut.flatten(), 0.0, 1.0).astype(np.float32)
        else:
            fut_norm = np.zeros(0, dtype=np.float32)

        cap_norm = np.array([np.clip(self.capacity_left_m / max(1e-9, self.weekly_capacity_m), 0.0, 1.0)], dtype=np.float32)
        last_norm = np.array([0.0 if self.last_idx < 0 else (self.last_idx + 1) / self.n_items], dtype=np.float32)
        setups_norm = np.array([np.tanh(self.setups_done / 10.0)], dtype=np.float32)

        return np.concatenate([inv_norm.astype(np.float32), fut_norm, cap_norm, last_norm, setups_norm], axis=0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.reset_state()
        return self._obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        idx_len, idx_lot = int(action[0]), int(action[1])
        idx_len = int(np.clip(idx_len, 0, self.n_items - 1))
        idx_lot = int(np.clip(idx_lot, 0, len(self.lot_bins_m) - 1))

        lot_m = float(self.lot_bins_m[idx_lot])  # metros totais puxados da(s) bobina(s)
        length_m = float(self.lengths_m[idx_len])

        # Verifica capacidade semanal e aplica clamp + penalidade se exceder
        over = max(0.0, lot_m - self.capacity_left_m)
        if over > 1e-9:
            lot_m = max(0.0, self.capacity_left_m)

        # Setup: troca de comprimento
        is_setup = (self.last_idx != -1 and self.last_idx != idx_len)
        if is_setup:
            self.setups_done += 1

        # Garante suprimento de bobina(s) para cobrir lot_m
        remaining_to_pull = lot_m
        while remaining_to_pull > self.coil_remaining_m + 1e-9:
            # consome o que resta desta bobina
            remaining_to_pull -= self.coil_remaining_m
            self.coil_remaining_m = 0.0
            # abre nova bobina (debita sucata fixa)
            self._open_new_coil()

        # consome o restante desta bobina
        self.coil_remaining_m -= remaining_to_pull
        self.coil_remaining_m = max(0.0, self.coil_remaining_m)

        # Produção de chapas e retalho de lote
        produced_sheets = int(math.floor(lot_m / max(1e-9, length_m)))
        produced_m = produced_sheets * length_m
        lot_remainder_m = max(0.0, lot_m - produced_m)  # sobra < 1 chapa
        waste_m = lot_remainder_m  # retalho de lote
        self.total_waste_m += waste_m

        # Consome capacidade (metros puxados)
        self.capacity_left_m -= lot_m
        self.capacity_left_m = max(0.0, self.capacity_left_m)

        # Atualiza inventário
        if produced_sheets > 0:
            self.inventory[idx_len] += produced_sheets
            self.total_produced_m += produced_m
            self.last_idx = idx_len

        # Vendas da semana corrente (sem backlog)
        sold_vec = np.minimum(self.inventory, self.current_dem)
        sold = int(sold_vec.sum())
        self.inventory -= sold_vec
        self.current_dem -= sold_vec
        self.total_sold += sold

        # Recompensa
        stock_units = float(self.inventory.sum())
        reward = -(
            self.w_waste * (waste_m) +
            self.w_stock * stock_units +
            self.w_setups * (1.0 if is_setup else 0.0) +
            self.w_viol * (1.0 if over > 1e-9 else 0.0)
        )

        # Terminação/rotação semanal
        terminated = False
        truncated = False
        self.total_steps += 1

        week_roll = (self.capacity_left_m <= 1e-9) or (self.current_dem.sum() == 0)
        if week_roll:
            self.week_idx += 1
            if self.week_idx >= self.horizon_weeks:
                terminated = True
            else:
                # nova semana: reseta capacidade e previsões; mantém bobina corrente
                self.capacity_left_m = self.weekly_capacity_m
                self._update_weekly_views()
                self.last_idx = -1  # "esquece" o último comprimento entre semanas

        info = {
            "produced_m": produced_m,
            "produced_sheets": produced_sheets,
            "sold": sold,
            "waste_m_step": waste_m,
            "waste_m_total": self.total_waste_m,
            "stock_units": stock_units,
            "setups_done": self.setups_done,
            "week_idx": self.week_idx,
            "coils_used": self.coils_used
        }

        return self._obs(), float(reward), terminated, truncated, info


# =========================
# Fábrica de ambiente (por YAML)
# =========================
def make_env_from_yaml(cfg_path: str | Path, seed: int = 42) -> CtlCuttingEnv:
    cfg = load_config(cfg_path)
    env = CtlCuttingEnv(
        lengths_mm=cfg["lengths_mm"],
        lot_bins_m=cfg["lot_bins_m"],
        weekly_minutes=cfg["weekly_minutes"],
        speed_m_per_min=cfg["speed_m_per_min"],
        horizon_weeks=cfg["horizon_weeks"],
        forecast_k=cfg["forecast_k"],
        coil_length_m=cfg["coil_length_m"],
        coil_fixed_scrap_frac=cfg["coil_fixed_scrap_frac"],
        demand_cfg=cfg["demand"],
        reward_weights=cfg["reward_weights"],
        seed=seed,
    )
    return env
