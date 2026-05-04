"""
Training script for AML Transaction Graph Generation
"""

import os
import sys
import torch
import numpy as np
import random
import json
import subprocess
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from types import SimpleNamespace

from models import GCNPolicy
from algorithms import PPO
from environment import TransactionEnvWrapper

# Add utils to path for S3 utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from s3_utils import ensure_dataset_files


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit():
    """Get current git commit hash if available"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_job_name():
    """Get HyperPod job name from environment if running remotely"""
    return os.environ.get('JOB_NAME', None)


@hydra.main(config_path="config", config_name="train_aml", version_base=None)
def train(cfg: DictConfig):
    """Main training loop"""
    # Track start time
    started_at = datetime.utcnow().isoformat() + 'Z'

    # Convert Hydra config to namespace for easy attribute access
    config = SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))
    print(config)

    # Add derived attributes if needed
    if not hasattr(config, 'name_full'):
        config.name_full = f"{config.env}_{config.dataset}_{config.name}"
    if not hasattr(config, 'name_full_load'):
        config.name_full_load = f"{config.env}_{config.dataset}_{config.name_load}_{config.load_step}"

    # Add fresh_csv support - generate unique name if requested
    if hasattr(config, 'fresh_csv') and config.fresh_csv:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        config.name_full = f"{config.name_full}_{timestamp}"
        print(f"Fresh CSV mode: output name updated to {config.name_full}")

    # Capture git commit and job name
    git_commit = get_git_commit()
    job_name = get_job_name()
    if git_commit:
        print(f"Git commit: {git_commit}")
    if job_name:
        print(f"HyperPod job: {job_name}")

    # Set device
    device = torch.device(config.device)

    # Set seed
    set_seed(config.seed)

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('generated_graphs', exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    # Ensure dataset files are available (download from S3 if needed)
    print("=" * 70)
    print("CHECKING DATASET FILES")
    print("=" * 70)
    data_dir = Path(config.dataset_path).parent
    required_files = [
        Path(config.dataset_path),
        Path(config.offsets_path),
        Path(config.index_path),
        Path(config.stats_path),
    ]

    missing_files = [f for f in required_files if not f.exists()]

    if missing_files:
        print(f"Missing {len(missing_files)} dataset files. Attempting to download from S3...")
        if not ensure_dataset_files(str(data_dir), verbose=True):
            print("\n× Failed to download dataset files from S3")
            print(f"\nMissing files:")
            for f in missing_files:
                print(f"  - {f}")
            print(f"\nEither:")
            print(f"  1. Run data prep: cd src/utils && python extract_sort.py")
            print(f"  2. Ensure AWS S3 access is configured")
            raise FileNotFoundError("Dataset files not found and could not be downloaded")

    # Verify all files now exist
    for f in required_files:
        if f.exists():
            size_mb = f.stat().st_size / (1024 ** 2)
            print(f"✓ {f.name} ({size_mb:.1f} MB)")
        else:
            raise FileNotFoundError(f"Required file not found: {f}")

    print("=" * 70)

    # Create environment
    print(f"Creating transaction graph environment")
    print(f"  Dataset: {config.dataset_path}")
    env = TransactionEnvWrapper(config)
    env.seed(config.seed)

    # Create policy
    print("Creating policy network")
    policy = GCNPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        atom_type_num=env.atom_type_num,
        config=config
    ).to(device)

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create PPO algorithm
    ppo = PPO(policy, config, device=device)

    # Load checkpoint if specified
    if config.load:
        checkpoint_path = f'checkpoints/{config.name_full_load}.pt'
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            ppo.load(checkpoint_path)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    # Save resolved runtime config
    resolved_config_path = f'generated_graphs/{config.name_full}.config.yaml'
    with open(resolved_config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"Saved resolved config to: {resolved_config_path}")

    # Open CSV for generated graphs
    csv_path = f'generated_graphs/{config.name_full}.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('iteration,nodes,edges,reward,avg_degree,timed_out,weakly_connected,num_components,cycles,depth,max_out,stop_reason\n')

    # Training loop
    print(f"Starting training for {config.num_steps} steps")
    print(f"  Expert training: iterations {config.expert_start}-{config.expert_end}")
    print(f"  RL training: iterations {config.rl_start}-{config.rl_end}")
    print(f"  Curriculum: {config.curriculum} ({config.curriculum_num} levels)")

    iteration = ppo.total_steps // config.timesteps_per_batch
    curriculum_level = 0

    while ppo.total_steps < config.num_steps:
        iteration += 1

        # Update curriculum level
        if config.curriculum:
            curriculum_level = min(
                iteration // config.curriculum_step,
                config.curriculum_num - 1
            )

        # Phase 1: Expert imitation
        if config.expert_start <= iteration <= config.expert_end:
            expert_stats = ppo.train_expert(
                env,
                config.optim_batchsize,
                curriculum=config.curriculum,
                level=curriculum_level,
                level_total=config.curriculum_num
            )

            # Log expert masking statistics every 100 iterations
            if iteration % 100 == 0:
                print(f"Expert Iter {iteration} | Level {curriculum_level} | "
                      f"Loss {expert_stats['expert_loss']:.4f} | "
                      f"Stop: {expert_stats['num_stop']}/{expert_stats['num_stop'] + expert_stats['num_node']} | "
                      f"L_stop {expert_stats['loss_stop']:.4f} | L_nodes {expert_stats['loss_nodes']:.4f}")

        # Phase 2: RL training
        if config.rl_start <= iteration <= config.rl_end:
            # Collect rollouts
            buffer, ep_info = ppo.collect_rollouts(env, config.timesteps_per_batch)

            # Update policy
            train_stats = ppo.update(buffer)

            # Log generated graphs
            if ep_info:
                with open(csv_path, 'a') as f:
                    for ep in ep_info:
                        if 'final_graph' in ep.get('info', {}):
                            G = ep['info']['final_graph']
                            n = G.number_of_nodes()
                            m = G.number_of_edges()
                            avg_deg = (2 * m / n) if n > 0 else 0
                            # Extract observability metrics
                            timed_out = int(ep['info'].get('timed_out', False))
                            weakly_connected = int(ep['info'].get('weakly_connected', False))
                            num_components = ep['info'].get('num_components', 0)
                            cycles = ep['info'].get('cycles', 0)
                            depth = ep['info'].get('depth', 0)
                            max_out = ep['info'].get('max_out', 0)
                            stop_reason = ep['info'].get('stop_reason', 'unknown')
                            f.write(f"{iteration},{n},{m},{ep['reward']:.3f},{avg_deg:.2f},"
                                    f"{timed_out},{weakly_connected},{num_components},"
                                    f"{cycles},{depth},{max_out},{stop_reason}\n")

            # Print progress
            if iteration % 10 == 0:
                avg_reward = np.mean([e['reward'] for e in ep_info]) if ep_info else 0.0
                avg_length = np.mean([e['length'] for e in ep_info]) if ep_info else 0.0
                print(f"Iter {iteration} | Steps {ppo.total_steps} | "
                      f"Episodes {ppo.total_episodes} | Level {curriculum_level} | "
                      f"Reward {avg_reward:.2f} | Len {avg_length:.1f}")

        # Save checkpoint
        if iteration % config.save_every == 0:
            checkpoint_path = f'checkpoints/{config.name_full}_{iteration}.pt'
            ppo.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Write iteration marker
        with open(csv_path, 'a') as f:
            f.write(f"# Iteration {iteration}, Level {curriculum_level}\n")

    # Save final checkpoint
    final_checkpoint_path = f'checkpoints/{config.name_full}_{iteration}.pt'
    if not os.path.exists(final_checkpoint_path):
        ppo.save(final_checkpoint_path)
        print(f"Saved final checkpoint to {final_checkpoint_path}")

    # Close environment
    env.close()

    # Track end time
    ended_at = datetime.utcnow().isoformat() + 'Z'

    # Prepare run summary
    run_summary = {
        'phase': 'phase1',
        'status': 'ok',
        'job_name': job_name,
        'git_commit': git_commit,
        'reward_type': getattr(config, 'reward_type', 'structural_smooth'),
        'generated_csv_path': csv_path,
        'resolved_config_path': resolved_config_path,
        'latest_checkpoint_path': final_checkpoint_path,
        'total_steps': ppo.total_steps,
        'total_episodes': ppo.total_episodes,
        'final_iteration': iteration,
        'curriculum': getattr(config, 'curriculum', False),
        'curriculum_step': getattr(config, 'curriculum_step', 50),
        'use_laundering_only': getattr(config, 'use_laundering_only', True),
        'seed': config.seed,
        'started_at': started_at,
        'ended_at': ended_at
    }

    # Save run summary
    run_summary_path = f'generated_graphs/{config.name_full}.run_summary.json'
    with open(run_summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Phase: {run_summary['phase']}")
    print(f"Status: {run_summary['status']}")
    print(f"Reward type: {run_summary['reward_type']}")
    print(f"Total steps: {run_summary['total_steps']:,}")
    print(f"Total episodes: {run_summary['total_episodes']:,}")
    print(f"Final iteration: {run_summary['final_iteration']}")
    print(f"Curriculum: {run_summary['curriculum']} (step={run_summary['curriculum_step']})")
    print(f"Laundering only: {run_summary['use_laundering_only']}")
    if git_commit:
        print(f"Git commit: {git_commit[:8]}")
    if job_name:
        print(f"Job name: {job_name}")
    print("\nArtifacts:")
    print(f"  CSV: {csv_path}")
    print(f"  Config: {resolved_config_path}")
    print(f"  Checkpoint: {final_checkpoint_path}")
    print(f"  Summary: {run_summary_path}")
    print("=" * 70)

    # Upload artifacts to S3 if running on HyperPod
    s3_artifact_root = os.environ.get('S3_ARTIFACT_ROOT')
    if s3_artifact_root and job_name:
        print("\n" + "=" * 70)
        print(f"Uploading artifacts to {s3_artifact_root}...")
        print("=" * 70)

        # Upload generated graphs
        try:
            cmd = f"aws s3 cp --recursive generated_graphs/ {s3_artifact_root}/generated_graphs/"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✓ Uploaded generated_graphs/")
            else:
                print(f"× Failed to upload generated_graphs/: {result.stderr}")
        except Exception as e:
            print(f"× Error uploading generated_graphs/: {e}")

        # Upload checkpoints
        try:
            cmd = f"aws s3 cp --recursive checkpoints/ {s3_artifact_root}/checkpoints/"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✓ Uploaded checkpoints/")
            else:
                print(f"× Failed to upload checkpoints/: {result.stderr}")
        except Exception as e:
            print(f"× Error uploading checkpoints/: {e}")

        print("=" * 70)


if __name__ == '__main__':
    train()
