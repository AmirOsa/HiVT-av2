import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from torch_geometric.data import DataLoader
import pytorch_lightning as pl

# ── Add repo to path ──────────────────────────────────────────────────────────
REPO_DIR = "/content/HiVT-av2"
sys.path.insert(0, REPO_DIR)

from datasets import ArgoverseV2Dataset
from models.hivt import HiVT

# ── Config ────────────────────────────────────────────────────────────────────
MF_DATA_ROOT  = "/content/drive/MyDrive/Amir_Dataset/HiVT-project/av2/motion-forecasting"
CKPT_PATH     = "/content/drive/MyDrive/Amir_Dataset/HiVT_av2_checkpoints/epoch=63-step=4032.ckpt"
OUTPUT_CSV    = "/content/drive/MyDrive/Amir_Dataset/HiVT-project_Confidence/hivt_predictions_inspection.csv"
BATCH_SIZE    = 1  # Keep at 1 so we can track per-scenario info easily
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

# Heading change thresholds (same as Nadeem's heuristic)
TURN_THRESH_DEG      = 20.0
LANE_CHANGE_THRESH_DEG = 5.0

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

# ── Helper: infer intention from trajectory ───────────────────────────────────
def infer_intention_from_trajectory(traj):
    """
    Given a trajectory of shape [60, 2] in local coordinates,
    infer the intention using the same thresholds as Nadeem's heuristic.
    """
    if traj.shape[0] < 2:
        return "OTHER"

    # Compute heading change from first to last point
    start = traj[0]
    end   = traj[-1]
    dx    = end[0] - start[0]
    dy    = end[1] - start[1]

    # Total displacement
    displacement = np.sqrt(dx**2 + dy**2)

    # Average speed (assuming 0.1s per step = 10Hz)
    speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1) / 0.1
    avg_speed = speeds.mean()

    # Stopped / Parked
    if avg_speed < 0.5:
        if displacement < 0.5:
            return "PARKED"
        else:
            return "STOPPING_STOPPED"

    # Compute heading change using atan2
    heading_change_rad = np.arctan2(dy, dx)
    heading_change_deg = np.degrees(heading_change_rad)

    if abs(heading_change_deg) > TURN_THRESH_DEG:
        return "TURN_LEFT" if heading_change_deg > 0 else "TURN_RIGHT"
    elif abs(heading_change_deg) > LANE_CHANGE_THRESH_DEG:
        return "LEFT_CHANGE_LANE" if heading_change_deg > 0 else "RIGHT_CHANGE_LANE"
    else:
        return "KEEP_LANE"

# ── Run inference ─────────────────────────────────────────────────────────────
print("\nRunning HiVT inference...")
rows = []

with torch.no_grad():
    for batch_idx, data in enumerate(loader):
        try:
            data = data.to(DEVICE)
            y_hat, pi = model(data)

            # y_hat: [6, N, 60, 2] — 6 modes, N agents, 60 steps, xy
            # pi:    [N, 6]        — mode probability logits per agent

            # Convert pi logits to probabilities
            probs = torch.softmax(pi, dim=-1)  # [N, 6]

            # Get scenario info
            seq_id   = data.seq_id if hasattr(data, 'seq_id') else f"scenario_{batch_idx}"
            if isinstance(seq_id, list):
                seq_id = seq_id[0]

            # Extract log_id from scenario_id (format: {log_id}_w{window_idx})
            parts  = seq_id.rsplit('_w', 1)
            log_id = parts[0] if len(parts) == 2 else seq_id

            # Get track_ids for all nodes in this scenario
            # We load the raw parquet to get track_ids
            scenario_parquet = None
            raw_dir = os.path.join(MF_DATA_ROOT, "dataset", "train", "scenarios", seq_id)
            parquet_files = list(Path(raw_dir).glob("*.parquet")) if os.path.exists(raw_dir) else []
            if parquet_files:
                scenario_parquet = pd.read_parquet(parquet_files[0])

            num_nodes    = data.num_nodes
            focal_index  = data.agent_index.item() if hasattr(data, 'agent_index') else 0

            # y_hat shape: [6, N, 60, 2]
            # For each agent, get their 6 trajectories and probabilities
            for node_idx in range(num_nodes):
                # Get track_id for this node
                track_id = "UNKNOWN"
                if scenario_parquet is not None:
                    track_ids_list = scenario_parquet['track_id'].unique().tolist()
                    if node_idx < len(track_ids_list):
                        track_id = track_ids_list[node_idx]

                is_focal = (node_idx == focal_index)

                # Get 6 trajectories for this agent [6, 60, 2]
                agent_trajs = y_hat[:, node_idx, :, :2].cpu().numpy()  # [6, 60, 2]
                agent_probs = probs[node_idx].cpu().numpy()             # [6]

                # Best mode (highest probability)
                best_mode     = agent_probs.argmax()
                best_traj     = agent_trajs[best_mode]   # [60, 2]
                best_prob     = agent_probs[best_mode]

                # Infer intention from best trajectory
                predicted_intention = infer_intention_from_trajectory(best_traj)

                # Infer intention for all 6 modes
                mode_intentions = [infer_intention_from_trajectory(agent_trajs[m]) for m in range(6)]

                rows.append({
                    'scenario_id'        : seq_id,
                    'log_id'             : log_id,
                    'track_id'           : track_id,
                    'is_focal_agent'     : is_focal,
                    'best_mode'          : int(best_mode),
                    'predicted_intention': predicted_intention,
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
        print(f"  {'track_id':<40} {'focal':<8} {'predicted_intention':<22} {'traj_confidence':<18} {'all_mode_intentions'}")
        print(f"  {'-'*120}")
        scen_df = log_df[log_df['scenario_id'] == scenario_id]
        for _, row in scen_df.iterrows():
            mode_str = " | ".join([f"m{i}:{row[f'mode_{i}_intention']}({row[f'mode_{i}_prob']:.2f})"
                                   for i in range(6)])
            print(f"  {row['track_id']:<40} {str(row['is_focal_agent']):<8} "
                  f"{row['predicted_intention']:<22} {row['traj_confidence']:.4f}           "
                  f"{mode_str}")

    # Log summary
    print(f"\n  Summary for {log_id}:")
    print(f"    Total agents     : {len(log_df)}")
    print(f"    Avg traj confidence: {log_df['traj_confidence'].mean():.4f}")
    print(f"    Intention breakdown:")
    for intent, count in log_df['predicted_intention'].value_counts().items():
        avg_conf = log_df[log_df['predicted_intention'] == intent]['traj_confidence'].mean()
        print(f"      {intent:<22}: {count}  (avg conf: {avg_conf:.4f})")

# ── Grand total ───────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("GRAND TOTAL — ALL SCENARIOS")
print("="*80)
print(f"  Total agents         : {len(df)}")
print(f"  Total scenarios      : {df['scenario_id'].nunique()}")
print(f"  Avg traj confidence  : {df['traj_confidence'].mean():.4f}")
print(f"\n  Intention breakdown:")
for intent, count in df['predicted_intention'].value_counts().items():
    avg_conf = df[df['predicted_intention'] == intent]['traj_confidence'].mean()
    print(f"    {intent:<22}: {count}  (avg conf: {avg_conf:.4f})")

# ── Save CSV ──────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ CSV saved to: {OUTPUT_CSV}")
