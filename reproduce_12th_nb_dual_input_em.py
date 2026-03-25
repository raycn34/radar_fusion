import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

DT = 0.1

@dataclass
class Config:
    traj_path: str = './0400pm-0415pm/trajectories-0400pm-0415pm.csv'
    signal_path: str = './0400pm-0415pm/Peachtree_12th_NB_0400-0415.csv'
    direction_code: int = 2   # NB
    movement_code: int = 1    # TH
    section_id: int = 3
    int_id: int = 3
    stopline_y: float = 1185.0
    region_entry_y: float = 985.0
    radar_band_low: float = 900.0
    radar_band_high: float = 1000.0
    speed_prior_scale: float = 0.25
    q0_band: int = 8
    startup_bounds: Tuple[float, float] = (0.0, 6.0)
    hsat_bounds: Tuple[float, float] = (0.5, 6.0)
    queue_speed_threshold: float = 5.0
    max_video_depth_ft: float = 200.0
    min_radar_samples: int = 10
    max_em_iter: int = 60
    tol: float = 1e-4


def load_signal(path: str) -> pd.DataFrame:
    sig = pd.read_csv(path)
    sig.columns = [c.strip() for c in sig.columns]
    need = ['BG_Thru', 'BY_Thru', 'BR_Thru']
    miss = [c for c in need if c not in sig.columns]
    if miss:
        raise ValueError(f'Missing signal columns: {miss}')
    sig = sig.sort_values('BG_Thru').reset_index(drop=True)
    return sig


def load_traj(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    req = ['Vehicle_ID','Frame_ID','Local_Y','v_Vel','Direction','Movement','Section_ID','Int_ID','v_Length']
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f'Missing trajectory columns: {miss}')
    return df.sort_values(['Vehicle_ID','Frame_ID']).reset_index(drop=True)


def crossing_frame(y: np.ndarray, f: np.ndarray, target: float) -> float:
    idx = np.where((y[:-1] < target) & (y[1:] >= target))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    y0, y1 = y[i], y[i+1]
    f0, f1 = f[i], f[i+1]
    if y1 == y0:
        return float(f0)
    r = (target - y0) / (y1 - y0)
    return float(f0 + r * (f1 - f0))


def extract_vehicle_observations(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    base = df[(df['Direction'] == cfg.direction_code) & (df['Movement'] == cfg.movement_code)].copy()
    for vid, g in base.groupby('Vehicle_ID'):
        g = g.sort_values('Frame_ID')
        y = g['Local_Y'].to_numpy()
        f = g['Frame_ID'].to_numpy()
        entry_200 = crossing_frame(y, f, cfg.region_entry_y)
        stopline = crossing_frame(y, f, cfg.stopline_y)
        # anonymous upstream radar speed from 900-1000 ft band
        rg = g[(g['Local_Y'] >= cfg.radar_band_low) & (g['Local_Y'] <= cfg.radar_band_high)]
        radar_mean = rg['v_Vel'].mean() if len(rg) > 0 else np.nan
        radar_n = len(rg)
        # whether this vehicle is plausibly queued at stopline start
        rows.append({
            'Vehicle_ID': vid,
            'entry_200_frame': entry_200,
            'stopline_frame': stopline,
            'travel_time_s': (stopline - entry_200) * DT if pd.notna(entry_200) and pd.notna(stopline) else np.nan,
            'radar_mean_speed': radar_mean,
            'radar_n': radar_n,
        })
    return pd.DataFrame(rows)


def assign_cycles(sig: pd.DataFrame, event_frame: float) -> Optional[int]:
    if pd.isna(event_frame):
        return None
    for j in range(len(sig) - 1):
        if sig.loc[j, 'BG_Thru'] <= event_frame < sig.loc[j+1, 'BG_Thru']:
            return j
    return None


def build_cycle_observations(veh_obs: pd.DataFrame, sig: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for j in range(len(sig) - 1):
        bg = float(sig.loc[j, 'BG_Thru'])
        by = float(sig.loc[j, 'BY_Thru'])
        br = float(sig.loc[j, 'BR_Thru'])
        next_bg = float(sig.loc[j+1, 'BG_Thru'])
        eff_green_s = (br - bg) * DT
        cyc_s = (next_bg - bg) * DT

        arr = veh_obs[(veh_obs['entry_200_frame'] >= bg) & (veh_obs['entry_200_frame'] < next_bg)].copy()
        dep = veh_obs[(veh_obs['stopline_frame'] >= bg) & (veh_obs['stopline_frame'] < br)].copy().sort_values('stopline_frame')
        dep_times = ((dep['stopline_frame'] - bg) * DT).tolist()
        entry_times = ((arr['entry_200_frame'] - bg) * DT).tolist()
        tt = arr['travel_time_s'].dropna().to_numpy()
        radar_vals = arr.loc[arr['radar_n'] >= cfg.min_radar_samples, 'radar_mean_speed'].dropna().to_numpy()
        rows.append({
            'cycle_id': j + 1,
            'BG_Thru': bg,
            'BR_Thru': br,
            'next_BG_Thru': next_bg,
            'cycle_length_s': cyc_s,
            'effective_green_s': eff_green_s,
            'entry_times_s': entry_times,
            'pass_times_s': dep_times,
            'n_video_entry': len(entry_times),
            'n_video_pass': len(dep_times),
            'mean_tt_video_s': float(np.mean(tt)) if len(tt) else np.nan,
            'radar_mean_speed': float(np.mean(radar_vals)) if len(radar_vals) else np.nan,
            'radar_n': int(len(radar_vals)),
        })
    out = pd.DataFrame(rows)
    # keep 9 valid cycles if present
    return out


def normal_pdf(x, mu, sd):
    sd = max(sd, 1e-6)
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def fit_speed_mixture(all_speeds: np.ndarray, max_iter: int = 100) -> Dict[str, float]:
    x = np.asarray(all_speeds)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {'pi': 0.8, 'mu_ff': 50.0, 'sd_ff': 4.0, 'mu_ct': 35.0, 'sd_ct': 8.0}
    mu_ff = np.percentile(x, 70)
    mu_ct = np.percentile(x, 30)
    sd_ff = np.std(x) if np.std(x) > 1 else 4.0
    sd_ct = max(sd_ff * 1.2, 6.0)
    pi = 0.7
    for _ in range(max_iter):
        p1 = pi * normal_pdf(x, mu_ff, sd_ff)
        p0 = (1 - pi) * normal_pdf(x, mu_ct, sd_ct)
        gamma = p1 / (p1 + p0 + 1e-12)
        pi = float(np.mean(gamma))
        mu_ff = float(np.sum(gamma * x) / (np.sum(gamma) + 1e-12))
        mu_ct = float(np.sum((1 - gamma) * x) / (np.sum(1 - gamma) + 1e-12))
        sd_ff = float(np.sqrt(np.sum(gamma * (x - mu_ff) ** 2) / (np.sum(gamma) + 1e-12)))
        sd_ct = float(np.sqrt(np.sum((1 - gamma) * (x - mu_ct) ** 2) / (np.sum(1 - gamma) + 1e-12)))
    return {'pi': pi, 'mu_ff': mu_ff, 'sd_ff': max(sd_ff, 1.0), 'mu_ct': mu_ct, 'sd_ct': max(sd_ct, 1.0)}


def posterior_ff_prob(x: np.ndarray, mix: Dict[str, float]) -> np.ndarray:
    p1 = mix['pi'] * normal_pdf(x, mix['mu_ff'], mix['sd_ff'])
    p0 = (1 - mix['pi']) * normal_pdf(x, mix['mu_ct'], mix['sd_ct'])
    return p1 / (p1 + p0 + 1e-12)


def weighted_prior_lambda(speed_mean: float, eff_green_s: float, distance_ft: float, cfg: Config, mix: Dict[str, float]) -> Tuple[float, float]:
    if not np.isfinite(speed_mean) or speed_mean <= 1e-6:
        return np.nan, np.nan
    gamma = posterior_ff_prob(np.array([speed_mean]), mix)[0]
    tff = distance_ft / max(speed_mean, 1e-6)
    # soft prior for arrivals over the cycle from anonymous speed only
    # slower speed implies lower effective arrival intensity into the 200 ft window
    lam = gamma * (1.0 / max(tff, 1e-6))
    sd = max(cfg.speed_prior_scale * lam, 0.05)
    return lam, sd


def estimate_tau_hsat_from_passes(pass_times_by_cycle: List[List[float]], max_rank: int = 6) -> Tuple[float, float]:
    xs, ys = [], []
    for times in pass_times_by_cycle:
        if len(times) < 3:
            continue
        times = np.array(sorted(times[:max_rank]))
        ranks = np.arange(1, len(times) + 1)
        xs.extend((ranks - 1).tolist())
        ys.extend(times.tolist())
    if len(xs) < 3:
        return 2.0, 2.0
    X = np.column_stack([np.ones(len(xs)), np.array(xs)])
    beta, *_ = np.linalg.lstsq(X, np.array(ys), rcond=None)
    tau, hsat = beta[0], beta[1]
    tau = float(np.clip(tau, 0.0, 6.0))
    hsat = float(np.clip(hsat, 0.5, 6.0))
    return tau, hsat


def infer_states(cyc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    all_speed = cyc['radar_mean_speed'].dropna().to_numpy()
    mix = fit_speed_mixture(all_speed)
    tau, hsat = estimate_tau_hsat_from_passes(cyc['pass_times_s'].tolist())
    s = 1.0 / hsat
    cap_per_green = lambda g: g / hsat

    lam = np.zeros(len(cyc))
    q0 = np.zeros(len(cyc), dtype=int)
    y = np.zeros(len(cyc))
    qe = np.zeros(len(cyc), dtype=int)
    t_on = np.zeros(len(cyc))

    prev_qe = 0
    distance_ft = cfg.stopline_y - cfg.region_entry_y

    for it in range(cfg.max_em_iter):
        tau_old, hsat_old = tau, hsat
        # E step / lambda prior update
        lam_prior = []
        for j, row in cyc.iterrows():
            lp, lsd = weighted_prior_lambda(row['radar_mean_speed'], row['effective_green_s'], distance_ft, cfg, mix)
            if np.isfinite(lp):
                lam[j] = lp
            elif row['cycle_length_s'] > 0:
                lam[j] = row['n_video_entry'] / row['cycle_length_s']
            lam_prior.append(lp)
        # M1 tau, hsat from queued subset approximation
        tau, hsat = estimate_tau_hsat_from_passes(cyc['pass_times_s'].tolist())
        s = 1.0 / hsat
        # M2 q0 discrete search with continuity and conservation
        prev_qe = 0
        for j, row in cyc.iterrows():
            g = row['effective_green_s']
            arr = lam[j] * row['cycle_length_s']
            dep_obs = row['n_video_pass']
            q0_lb = max(0, dep_obs - int(np.floor(cap_per_green(g))))
            cand = range(q0_lb, q0_lb + cfg.q0_band + 1)
            best = None
            for q in cand:
                yj = min(q + arr, cap_per_green(g))
                qej = max(0.0, q + arr - yj)
                obj = (qej - prev_qe) ** 2
                if best is None or obj < best[0]:
                    best = (obj, q, yj, qej)
            _, q0j, yj, qej = best
            q0[j] = int(round(q0j))
            y[j] = yj
            qe[j] = int(round(qej))
            prev_qe = qej
            pass_times = row['pass_times_s']
            t_on[j] = pass_times[0] - tau if len(pass_times) else 0.0
        if abs(tau - tau_old) + abs(hsat - hsat_old) < cfg.tol:
            break

    out = cyc.copy()
    out['lambda_hat_vps'] = lam
    out['Q0_hat'] = q0
    out['y_hat'] = y
    out['Qe_hat'] = qe
    out['tau_hat_s'] = tau
    out['hsat_hat_s'] = hsat
    out['sat_flow_hat_vps'] = 1.0 / hsat
    out['sat_flow_hat_vph'] = 3600.0 / hsat
    out['t_on_hat_s'] = t_on
    return out


def summarize_average(est: pd.DataFrame) -> pd.DataFrame:
    cols = ['lambda_hat_vps','Q0_hat','y_hat','Qe_hat','tau_hat_s','hsat_hat_s','sat_flow_hat_vph','mean_tt_video_s']
    avg = est[cols].mean(numeric_only=True).to_frame(name='average_over_cycles').reset_index().rename(columns={'index':'parameter'})
    return avg


def main(cfg: Config = Config()):
    traj = load_traj(cfg.traj_path)
    sig = load_signal(cfg.signal_path)
    veh = extract_vehicle_observations(traj, cfg)
    cyc = build_cycle_observations(veh, sig, cfg)
    est = infer_states(cyc, cfg)
    avg = summarize_average(est)
    print(est[['cycle_id','n_video_entry','n_video_pass','radar_mean_speed','lambda_hat_vps','Q0_hat','y_hat','Qe_hat','tau_hat_s','hsat_hat_s','sat_flow_hat_vph','mean_tt_video_s']])
    print('\nAverages\n', avg)
    est.to_csv('cycle_estimates_dual_input_12th_NB.csv', index=False)
    avg.to_csv('cycle_estimates_dual_input_12th_NB_average.csv', index=False)


if __name__ == '__main__':
    main()
