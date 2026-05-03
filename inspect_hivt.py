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

# ── Config ────────────────────────────────────────────────────────────────────
MF_DATA_ROOT = "/content/drive/MyDrive/Amir_Dataset/HiVT-project_SMF/av2/motion-forecasting"
CKPT_PATH    = "/content/drive/MyDrive/Amir_Dataset/HiVT_av2_checkpoints/epoch=63-step=4032.ckpt"
OUTPUT_CSV   = "/content/drive/MyDrive/Amir_Dataset/HiVT-project_Confidence/hivt_predictions_inspection.csv"
BATCH_SIZE   = 1
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# Heading change thresholds (same as Nadeem's heuristic)
TURN_THRESH_DEG       = 20.0
LANE_CHANGE_THRESH_DEG = 5.0

# ── Helper: infer intention from trajectory ───────────────────────────────────
def infer_intention_from_trajectory(traj):
    """
    Given a trajectory of shape [N, 2] in local coordinates,
    infer the intention using the same thresholds as Nadeem's heuristic.
    """
    if traj.shape[0] < 2:
        return "OTHER"

    # Average speed (0.1s per step = 10Hz)
    speeds    = np.linalg.norm(np.diff(traj, axis=0), axis=1) / 0.1
    avg_speed = speeds.mean()

    # Total displacement
    displacement = np.linalg.norm(traj[-1] - traj[0])

    # Stopped / Parked
    if avg_speed < 0.5:
        return "PARKED" if displacement < 0.5 else "STOPPING_STOPPED"

    # Compute heading change using first and last few points for stability
    start_vec = traj[min(5, len(traj)-1)] - traj[0]
    end_vec   = traj[-1] - traj[max(0, len(traj)-6)]

    initial_heading = np.arctan2(start_vec[1], start_vec[0])
    final_heading   = np.arctan2(end_vec[1], end_vec[0])

    # Normalize to [-180, 180]
    heading_change_deg = np.degrees(final_heading - initial_heading)
    heading_change_deg = (heading_change_deg + 180) % 360 - 180

    if abs(heading_change_deg) > TURN_THRESH_DEG:
        return "TURN_LEFT" if heading_change_deg > 0 else "TURN_RIGHT"
    elif abs(heading_change_deg) > LANE_CHANGE_THRESH_DEG:
        return "LEFT_CHANGE_LANE" if heading_change_deg > 0 else "RIGHT_CHANGE_LANE"
    else:
        return "KEEP_LANE"

# ── Helper: get ground truth intention from parquet future steps ──────────────
def get_ground_truth_intention(scenario_parquet, track_id):
    """
    Compute ground truth intention from actual future trajectory in parquet.
    Uses timesteps 50-109 (future steps) for the given track_id.
    """
    if scenario_parquet is None:
        return "UNKNOWN"

    agent_df = scenario_parquet[
        (scenario_parquet['track_id'] == track_id) &
        (scenario_parquet['timestep'] >= 50)
    ].sort_values('timestep')

    if len(agent_df) < 5:
        return "UNKNOWN"

    traj = agent_df[['position_x', 'position_y']].to_numpy()
    return infer_intention_from_trajectory(traj)

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

# ── Run inference ─────────────────────────────────────────────────────────────
print("\nRunning HiVT inference...")
rows = []

with torch.no_grad():
    for batch_idx, data in enumerate(loader):
        try:
            data = data.to(DEVICE)
            y_hat, pi = model(data)

            # y_hat: [6, N, 60, 2]
            # pi:    [N, 6]
            probs = torch.softmax(pi, dim=-1)  # [N, 6]

            # Get scenario info
            seq_id = data.seq_id if hasattr(data, 'seq_id') else f"scenario_{batch_idx}"
            if isinstance(seq_id, list):
                seq_id = seq_id[0]

            # Extract log_id from scenario_id (format: {log_id}_w{window_idx})
            parts  = seq_id.rsplit('_w', 1)
            log_id = parts[0] if len(parts) == 2 else seq_id

            # Load raw parquet to get track_ids and ground truth
            scenario_parquet = None
            raw_dir = os.path.join(
                MF_DATA_ROOT, "dataset", "train", "scenarios", seq_id
            )
            parquet_files = list(Path(raw_dir).glob("*.parquet")) if os.path.exists(raw_dir) else []
            if parquet_files:
                scenario_parquet = pd.read_parquet(parquet_files[0])

            # Get ordered track_ids matching node indices
            track_ids_ordered = []
            if scenario_parquet is not None:
                # Get track_ids in the order they appear as unique values
                # This matches how ArgoverseV2Dataset builds track_id_to_node_idx
                track_ids_ordered = list(scenario_parquet.groupby('track_id').first().index)

            num_nodes   = data.num_nodes
            focal_index = data.agent_index.item() if hasattr(data, 'agent_index') else 0

            for node_idx in range(num_nodes):
                # Get track_id for this node
                track_id = track_ids_ordered[node_idx] if node_idx < len(track_ids_ordered) else "UNKNOWN"
                is_focal = (node_idx == focal_index)

                # Get 6 trajectories and probabilities for this agent
                agent_trajs = y_hat[:, node_idx, :, :2].cpu().numpy()  # [6, 60, 2]
                agent_probs = probs[node_idx].cpu().numpy()             # [6]

                # Best mode
                best_mode = agent_probs.argmax()
                best_traj = agent_trajs[best_mode]  # [60, 2]
                best_prob = agent_probs[best_mode]

                # Predicted intention from best trajectory
                predicted_intention = infer_intention_from_trajectory(best_traj)

                # Ground truth intention from actual future trajectory
                actual_intention = get_ground_truth_intention(scenario_parquet, track_id)
                correct = (predicted_intention == actual_intention) if actual_intention != "UNKNOWN" else None

                # All 6 mode intentions
                mode_intentions = [
                    infer_intention_from_trajectory(agent_trajs[m]) for m in range(6)
                ]

                rows.append({
                    'scenario_id'        : seq_id,
                    'log_id'             : log_id,
                    'track_id'           : track_id,
                    'is_focal_agent'     : is_focal,
                    'best_mode'          : int(best_mode),
                    'predicted_intention': predicted_intention,
                    'actual_intention'   : actual_intention,
                    'correct'            : correct,
                    'traj_confidence'    : float(best_prob),
                    'mode_0_prob'        : float(agent_probs[0]),
                    'mode_1_prob'        : float(agent_probs[1]),
                    'mode_2_prob'        : float(agent_probs[2]),
                    'mode_3_prob'        : float(agent_probs[3]),
                    'mode_4_prob'        : float(agent_probs[4]),
                    'mode_5_prob'        : float(agent_probs[5]),
                    'mode_0_intention'   : mode_intentions[0],
                    'mode_1_intention'   : mode_intentions[1],
                    'mode_2_intention'   : mode_intentions[2],
                    'mode_3_intention'   : mode_intentions[3],
                    'mode_4_intention'   : mode_intentions[4],
                    'mode_5_intention'   : mode_intentions[5],
                })

        except Exception as e:
            print(f"❌ Error on scenario {batch_idx}: {e}")
            continue

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

# ── Helper: print summary for a subset ───────────────────────────────────────
def print_summary(subset, label):
    known    = subset[subset['actual_intention'] != 'UNKNOWN']
    n_total  = len(known)
    n_correct = known['correct'].sum()
    avg_conf  = known['traj_confidence'].mean() if n_total > 0 else 0.0

    print(f"    Total agents       : {len(subset)}")
    print(f"    With ground truth  : {n_total}")
    print(f"    Correct intention  : {n_correct} / {n_total} ({100*n_correct/max(n_total,1):.1f}%)")
    print(f"    Avg traj confidence: {avg_conf:.4f}")
    print(f"    Per-intention accuracy:")
    for intent in sorted(known['actual_intention'].unique()):
        sub      = known[known['actual_intention'] == intent]
        acc      = sub['correct'].mean() * 100
        avg_ic   = sub['traj_confidence'].mean()
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
        print(f"  {'track_id':<40} {'focal':<8} {'predicted':<22} {'actual':<22} {'correct':<8} {'conf':<8} {'all modes'}")
        print(f"  {'-'*150}")
        scen_df = log_df[log_df['scenario_id'] == scenario_id]
        for _, row in scen_df.iterrows():
            mode_str = " | ".join([
                f"m{i}:{row[f'mode_{i}_intention'][:4]}({row[f'mode_{i}_prob']:.2f})"
                for i in range(6)
            ])
            correct_str = str(row['correct']) if row['correct'] is not None else "N/A"
            print(f"  {str(row['track_id']):<40} {str(row['is_focal_agent']):<8} "
                  f"{row['predicted_intention']:<22} {row['actual_intention']:<22} "
                  f"{correct_str:<8} {row['traj_confidence']:.4f}   {mode_str}")

    print(f"\n  Summary for {log_id}:")
    print_summary(log_df, log_id)

# ── Grand total ───────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("GRAND TOTAL — ALL SCENARIOS")
print("="*80)
print_summary(df, "ALL")

# ── Save CSV ──────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ CSV saved to: {OUTPUT_CSV}")
