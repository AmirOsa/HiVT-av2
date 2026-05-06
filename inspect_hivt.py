import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import DataLoader

# ── Add repo to path ──────────────────────────────────────────────────────────
REPO_DIR = "/content/HiVT-av2"
sys.path.insert(0, REPO_DIR)

from datasets import ArgoverseV2Dataset
from models.hivt import HiVT
from av2.map.map_api import ArgoverseStaticMap

# ── Config ────────────────────────────────────────────────────────────────────
MF_DATA_ROOT = "/content/drive/MyDrive/Amir_Dataset/HiVT-project_SMF/av2/motion-forecasting"
CKPT_PATH    = "/content/drive/MyDrive/Amir_Dataset/HiVT_av2_checkpoints/epoch=63-step=4032.ckpt"
OUTPUT_CSV   = "/content/drive/MyDrive/Amir_Dataset/HiVT-project_Confidence/hivt_focal_inspection.csv"
BATCH_SIZE   = 1
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Constants matching Nadeem's heuristic ─────────────────────────────────────
MIN_SPEED_STOPPED               = 0.5   # m/s — below this: PARKED or STOPPING
MIN_SPEED_MOVING                = 1.0   # m/s — FIX 1: turns only assigned above this
PARKED_MAX_DISP_M               = 0.5   # meters
HEADING_CHANGE_THRESH_TURN      = 20.0  # degrees
HEADING_CHANGE_THRESH_LANE_KEEP = 5.0   # degrees
KEEP_LANE_MAX_LAT_DIST          = 0.5   # meters — FIX 3: lateral displacement check

# ── Helper: load map for a scenario ──────────────────────────────────────────
def load_static_map(raw_dir):
    """Load ArgoverseStaticMap from scenario folder. Returns None if not found."""
    map_files = list(Path(raw_dir).glob("log_map_archive_*.json"))
    if not map_files:
        return None
    try:
        return ArgoverseStaticMap.from_json(map_files[0])
    except Exception as e:
        print(f"  ⚠️  Could not load map: {e}")
        return None

# ── Helper: check if a city-frame position is inside an intersection ──────────
def is_in_intersection(city_pos_xy, static_map):
    """
    Returns True if the given [x, y] city-frame position falls inside
    any intersection lane segment in the map.
    """
    if static_map is None or city_pos_xy is None:
        return False
    try:
        import shapely.geometry as shp
        point = shp.Point(city_pos_xy[0], city_pos_xy[1])
        for ls in static_map.get_scenario_lane_segments():
            if ls.is_intersection:
                polygon = shp.Polygon(ls.polygon_boundary[:, :2])
                if polygon.contains(point):
                    return True
        return False
    except Exception:
        return False

# ── Helper: infer intention from trajectory ───────────────────────────────────
def infer_intention_from_trajectory(traj, city_pos_at_t49=None, static_map=None):
    """
    Replicates Nadeem's get_vehicle_intention_heuristic_enhanced.

    traj            : [N, 2] agent-centric local frame, up to 60 steps
    city_pos_at_t49 : [2] city-frame position at timestep 49, used for
                      intersection check (FIX 2). Pass None to skip.
    static_map      : ArgoverseStaticMap instance (FIX 2). Pass None to skip.
    """
    if traj.shape[0] < 2:
        return "OTHER"

    # ── Speed and displacement ────────────────────────────────────────────────
    diffs        = np.diff(traj, axis=0)
    step_dists   = np.linalg.norm(diffs, axis=1)
    speeds       = step_dists / 0.1          # convert to m/s at 10 Hz
    avg_speed    = speeds.mean()
    displacement = np.linalg.norm(traj[-1] - traj[0])

    # ── PARKED / STOPPING_STOPPED ─────────────────────────────────────────────
    if avg_speed < MIN_SPEED_STOPPED:
        if displacement < PARKED_MAX_DISP_M:
            return "PARKED"
        else:
            return "STOPPING_STOPPED"

    # ── Heading change ────────────────────────────────────────────────────────
    n         = len(traj)
    start_vec = traj[min(5, n-1)] - traj[0]
    end_vec   = traj[-1] - traj[max(0, n-6)]

    if np.linalg.norm(start_vec) < 0.1 or np.linalg.norm(end_vec) < 0.1:
        return "KEEP_LANE"

    initial_heading    = np.arctan2(start_vec[1], start_vec[0])
    final_heading      = np.arctan2(end_vec[1], end_vec[0])
    heading_change_deg = np.degrees(final_heading - initial_heading)
    heading_change_deg = (heading_change_deg + 180) % 360 - 180  # [-180, 180]

    # ── TURN LEFT / TURN RIGHT ────────────────────────────────────────────────
    # FIX 1: only assign turn if avg_speed > MIN_SPEED_MOVING (1.0 m/s)
    if avg_speed > MIN_SPEED_MOVING and abs(heading_change_deg) > HEADING_CHANGE_THRESH_TURN:
        return "TURN_LEFT" if heading_change_deg > 0 else "TURN_RIGHT"

    # ── LEFT / RIGHT CHANGE LANE ──────────────────────────────────────────────
    # FIX 2: skip lane change label if agent is inside an intersection
    # (Nadeem uses map context to exclude intersection maneuvers from lane change)
    if (abs(heading_change_deg) > HEADING_CHANGE_THRESH_LANE_KEEP and
            abs(heading_change_deg) <= HEADING_CHANGE_THRESH_TURN):

        in_intersection = is_in_intersection(city_pos_at_t49, static_map)

        if not in_intersection:
            return "LEFT_CHANGE_LANE" if heading_change_deg > 0 else "RIGHT_CHANGE_LANE"
        # if in intersection, fall through to KEEP_LANE / OTHER below

    # ── KEEP_LANE ─────────────────────────────────────────────────────────────
    # FIX 3: lateral displacement must stay within lane width (0.5m)
    if abs(heading_change_deg) <= HEADING_CHANGE_THRESH_LANE_KEEP:
        direction = traj[-1] - traj[0]
        dir_norm  = np.linalg.norm(direction)
        if dir_norm > 0.01:
            direction_unit = direction / dir_norm
            perp           = np.array([-direction_unit[1], direction_unit[0]])
            lateral_devs   = np.abs((traj - traj[0]) @ perp)
            max_lat_dev    = lateral_devs.max()
        else:
            max_lat_dev = 0.0

        if max_lat_dev <= KEEP_LANE_MAX_LAT_DIST:
            return "KEEP_LANE"

    return "OTHER"

# ── Helper: get ground truth intention from parquet future steps ──────────────
def get_ground_truth_intention(scenario_parquet, track_id, static_map):
    if scenario_parquet is None:
        return "UNKNOWN"

    agent_df = scenario_parquet[
        (scenario_parquet['track_id'] == track_id) &
        (scenario_parquet['timestep'] >= 50)
    ].sort_values('timestep')

    if len(agent_df) < 5:
        return "UNKNOWN"

    traj = agent_df[['position_x', 'position_y']].to_numpy()

    # Get origin and heading from timestep 49 (last observed step)
    obs_df = scenario_parquet[
        (scenario_parquet['track_id'] == track_id) &
        (scenario_parquet['timestep'] == 49)
    ]
    if len(obs_df) == 0:
        return "UNKNOWN"

    heading = float(obs_df['heading'].iloc[0])
    origin  = np.array([
        float(obs_df['position_x'].iloc[0]),
        float(obs_df['position_y'].iloc[0])
    ])

    # City-frame position at t49 for intersection check
    city_pos_at_t49 = origin.copy()

    # Mirror exactly what process_argoverse_v2 does:
    # local_pos = (pos - origin) @ rotate_mat
    cos_h      = np.cos(heading)
    sin_h      = np.sin(heading)
    rotate_mat = np.array([
        [ cos_h, -sin_h],
        [ sin_h,  cos_h]
    ])
    traj = (traj - origin) @ rotate_mat

    return infer_intention_from_trajectory(traj, city_pos_at_t49, static_map)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading HiVT model...")
model = HiVT.load_from_checkpoint(
    checkpoint_path=CKPT_PATH,
    map_location=DEVICE,
    strict=False
)
model.eval()
model = model.to(DEVICE)
print("✅ HiVT model loaded")

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading converted MF dataset...")
dataset = ArgoverseV2Dataset(
    root=MF_DATA_ROOT,
    split='val',
    local_radius=model.hparams.local_radius
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)
print(f"✅ {len(dataset)} scenarios loaded")

# ── Run inference — focal agent only ─────────────────────────────────────────
print("\nRunning HiVT inference (focal agent only)...")
rows = []

with torch.no_grad():
    for batch_idx, data in enumerate(loader):
        try:
            data = data.to(DEVICE)
            y_hat, pi = model(data)

            # y_hat: [6, N, 60, 2]
            # pi:    [N, 6]
            probs = torch.softmax(pi, dim=-1)  # [N, 6]

            # ── Scenario info ─────────────────────────────────────────────────
            seq_id = data.seq_id if hasattr(data, 'seq_id') else f"scenario_{batch_idx}"
            if isinstance(seq_id, list):
                seq_id = seq_id[0]

            parts  = seq_id.rsplit('_w', 1)
            log_id = parts[0] if len(parts) == 2 else seq_id

            # ── Load raw parquet ──────────────────────────────────────────────
            scenario_parquet = None
            raw_dir = os.path.join(
                MF_DATA_ROOT, "dataset", "train", "scenarios", seq_id
            )
            parquet_files = list(Path(raw_dir).glob("*.parquet")) if os.path.exists(raw_dir) else []
            if parquet_files:
                scenario_parquet = pd.read_parquet(parquet_files[0])

            # ── Load map for intersection check (FIX 2) ───────────────────────
            static_map = load_static_map(raw_dir)

            # ── Focal track id directly from parquet ──────────────────────────
            focal_track_id = "UNKNOWN"
            if scenario_parquet is not None:
                focal_ids = scenario_parquet['focal_track_id'].unique()
                if len(focal_ids) > 0:
                    focal_track_id = focal_ids[0]

            # ── Focal agent trajectories and probabilities ────────────────────
            focal_index = data.agent_index.item() if hasattr(data, 'agent_index') else 0
            focal_trajs = y_hat[:, focal_index, :, :2].cpu().numpy()  # [6, 60, 2]
            focal_probs = probs[focal_index].cpu().numpy()             # [6]

            best_mode = focal_probs.argmax()
            best_traj = focal_trajs[best_mode]
            best_prob = focal_probs[best_mode]

            # ── City-frame position at t49 for intersection check ─────────────
            city_pos_at_t49 = None
            if scenario_parquet is not None and focal_track_id != "UNKNOWN":
                obs_df = scenario_parquet[
                    (scenario_parquet['track_id'] == focal_track_id) &
                    (scenario_parquet['timestep'] == 49)
                ]
                if len(obs_df) > 0:
                    city_pos_at_t49 = np.array([
                        float(obs_df['position_x'].iloc[0]),
                        float(obs_df['position_y'].iloc[0])
                    ])

            # ── Predicted intention ───────────────────────────────────────────
            predicted_intention = infer_intention_from_trajectory(
                best_traj, city_pos_at_t49, static_map
            )

            # ── Ground truth intention ────────────────────────────────────────
            actual_intention = get_ground_truth_intention(
                scenario_parquet, focal_track_id, static_map
            )
            correct = (predicted_intention == actual_intention) \
                if actual_intention != "UNKNOWN" else None

            # ── All 6 mode intentions ─────────────────────────────────────────
            mode_intentions = [
                infer_intention_from_trajectory(focal_trajs[m], city_pos_at_t49, static_map)
                for m in range(6)
            ]

            # ── Focal agent object_category ───────────────────────────────────
            focal_category = 3
            if scenario_parquet is not None and focal_track_id in scenario_parquet['track_id'].values:
                focal_category = int(scenario_parquet[
                    scenario_parquet['track_id'] == focal_track_id
                ]['object_category'].iloc[0])

            rows.append({
                'scenario_id'        : seq_id,
                'log_id'             : log_id,
                'track_id'           : focal_track_id,
                'object_category'    : focal_category,
                'is_focal_agent'     : True,
                'best_mode'          : int(best_mode),
                'predicted_intention': predicted_intention,
                'actual_intention'   : actual_intention,
                'correct'            : correct,
                'traj_confidence'    : float(best_prob),
                'mode_0_prob'        : float(focal_probs[0]),
                'mode_1_prob'        : float(focal_probs[1]),
                'mode_2_prob'        : float(focal_probs[2]),
                'mode_3_prob'        : float(focal_probs[3]),
                'mode_4_prob'        : float(focal_probs[4]),
                'mode_5_prob'        : float(focal_probs[5]),
                'mode_0_intention'   : mode_intentions[0],
                'mode_1_intention'   : mode_intentions[1],
                'mode_2_intention'   : mode_intentions[2],
                'mode_3_intention'   : mode_intentions[3],
                'mode_4_intention'   : mode_intentions[4],
                'mode_5_intention'   : mode_intentions[5],
            })

        except Exception as e:
            print(f"❌ Error on scenario {batch_idx}: {e}")
            import traceback; traceback.print_exc()
            continue

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

# ── Helper: print summary ─────────────────────────────────────────────────────
def print_summary(subset):
    known     = subset[subset['actual_intention'] != 'UNKNOWN']
    n_total   = len(known)
    n_correct = known['correct'].sum()
    avg_conf  = known['traj_confidence'].mean() if n_total > 0 else 0.0

    print(f"    Total focal agents : {len(subset)}")
    print(f"    With ground truth  : {n_total}")
    print(f"    Correct intention  : {n_correct} / {n_total} ({100*n_correct/max(n_total,1):.1f}%)")
    print(f"    Avg traj confidence: {avg_conf:.4f}")
    print(f"    Per-intention accuracy:")
    for intent in sorted(known['actual_intention'].unique()):
        sub    = known[known['actual_intention'] == intent]
        acc    = sub['correct'].mean() * 100
        avg_ic = sub['traj_confidence'].mean()
        print(f"      {intent:<22}: {acc:.1f}%  ({sub['correct'].sum()}/{len(sub)})  "
              f"avg conf: {avg_ic:.4f}")

# ── Print per log ─────────────────────────────────────────────────────────────
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

for log_id in df['log_id'].unique():
    print("\n" + "="*80)
    print(f"LOG: {log_id}")
    print("="*80)
    log_df = df[df['log_id'] == log_id]

    for scenario_id in log_df['scenario_id'].unique():
        print(f"\n  Scenario: {scenario_id}")
        print(f"  {'track_id':<40} {'predicted':<22} {'actual':<22} {'correct':<8} {'conf':<8} {'all modes'}")
        print(f"  {'-'*150}")
        scen_df = log_df[log_df['scenario_id'] == scenario_id]
        for _, row in scen_df.iterrows():
            mode_str = " | ".join([
                f"m{i}:{row[f'mode_{i}_intention'][:4]}({row[f'mode_{i}_prob']:.2f})"
                for i in range(6)
            ])
            correct_str = str(row['correct']) if row['correct'] is not None else "N/A"
            print(f"  {str(row['track_id']):<40} "
                  f"{row['predicted_intention']:<22} {row['actual_intention']:<22} "
                  f"{correct_str:<8} {row['traj_confidence']:.4f}   {mode_str}")

    print(f"\n  Summary for {log_id}:")
    print_summary(log_df)

# ── Grand total ───────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("GRAND TOTAL — ALL SCENARIOS (FOCAL AGENT ONLY)")
print("="*80)
print_summary(df)

# ── Save CSV ──────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ CSV saved to: {OUTPUT_CSV}")
