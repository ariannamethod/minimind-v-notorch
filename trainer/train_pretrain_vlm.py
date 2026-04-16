"""
MiniMind-V VLM pretraining — backed by notorch + Chuck optimizer

No PyTorch. No torch.distributed. No CUDA.
Just notorch (ctypes to libnotorch) and Chuck.

Trains vision encoder (patch embedding + projector) + text decoder
on image+text pairs from parquet dataset.

usage:
    python -m trainer.train_pretrain_vlm                              # train with defaults
    python -m trainer.train_pretrain_vlm --data_path dataset/i2t.parquet  # custom dataset
    python -m trainer.train_pretrain_vlm --resume weights/vlm.bin     # resume training
"""

import os
import sys
import time
import math
import json
import random
import argparse
import io

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_vlm import MiniMindVLM, VLMConfig
from ariannamethod.notorch_nn import seed as nt_seed


def load_parquet_dataset(path):
    """Load image+text pairs from parquet file. Returns list of (conversations, image_bytes)."""
    import pyarrow.parquet as pq
    import pyarrow as pa
    table = pa.Table.from_batches(pq.ParquetFile(path).iter_batches())
    data = []
    for i in range(len(table)):
        conversations = json.loads(table['conversations'][i].as_py())
        image_bytes = table['image_bytes'][i].as_py()
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        data.append((conversations, image_bytes))
    return data


def simple_tokenize(text, vocab_size=6400):
    """Simple character-level tokenization (fallback when no tokenizer available)."""
    return [min(ord(c), vocab_size - 1) for c in text]


def build_training_pair(conversations, image_token, image_token_len, max_len, tokenizer=None):
    """
    Build token_ids and target_ids from conversation data.
    Returns (token_ids, target_ids) as lists of int.
    """
    # Build prompt from conversations
    image_pad = image_token * image_token_len
    parts = []
    for turn in conversations:
        role = turn.get('role', 'user')
        content = turn.get('content', '')
        content = content.replace('<image>', image_pad)
        parts.append(f"{role}: {content}")
    text = '\n'.join(parts)

    if tokenizer:
        token_ids = tokenizer(text).input_ids[:max_len]
    else:
        token_ids = simple_tokenize(text, 6400)[:max_len]

    # Pad to max_len
    pad_id = 0
    if len(token_ids) < max_len:
        token_ids += [pad_id] * (max_len - len(token_ids))

    # Target: shifted by 1 (next-token prediction)
    target_ids = token_ids[1:] + [pad_id]

    return token_ids, target_ids


def get_lr(current_step, total_steps, lr, warmup_ratio=0.1):
    """Cosine schedule with linear warmup."""
    warmup = int(total_steps * warmup_ratio)
    min_lr = lr * 0.1
    if current_step < warmup:
        return lr * current_step / max(1, warmup)
    progress = (current_step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description='MiniMind-V VLM Pretrain (notorch)')
    parser.add_argument('--data_path', type=str, default='../dataset/pretrain_i2t.parquet')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=360)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--image_token_len', type=int, default=64)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='../out')
    parser.add_argument('--save_weight', type=str, default='pretrain_vlm')
    args = parser.parse_args()

    print("════════════════════════════════════════════════════════")
    print("  MiniMind-V — VLM pretraining (Python + libnotorch)")
    print("  no torch. no transformers. Chuck optimizer.")
    print("════════════════════════════════════════════════════════")

    # Config
    vlm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        image_size=args.image_size,
        patch_size=args.patch_size,
        image_token_len=args.image_token_len,
    )

    nt_seed(42)
    model = MiniMindVLM(vlm_config)
    n_params = model.count_params()
    print(f"model: {n_params:,} params (dim={args.hidden_size} L={args.num_hidden_layers})")
    print(f"vision: {vlm_config.image_size}px, patch={vlm_config.patch_size}, "
          f"n_patches={vlm_config.n_patches}, tokens={vlm_config.image_token_len}")

    # Load tokenizer if available
    tokenizer = None
    tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'model')
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"tokenizer: loaded from {tokenizer_path}")
    except Exception:
        print("tokenizer: using character-level fallback")

    # Load dataset
    if not os.path.exists(args.data_path):
        print(f"dataset not found: {args.data_path}")
        print("generating synthetic training data for testing...")
        dataset = None
    else:
        dataset = load_parquet_dataset(args.data_path)
        print(f"dataset: {args.data_path} ({len(dataset)} samples)")

    # Resume
    start_step = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        if model.load_weights(args.resume):
            print(f"resumed from {args.resume}")
            meta = args.resume + '.meta'
            if os.path.exists(meta):
                with open(meta) as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        start_step = int(lines[0].strip())
                        start_epoch = int(lines[1].strip()) if len(lines) > 1 else 0
                print(f"  start_epoch={start_epoch}, start_step={start_step}")

    total_steps = (len(dataset) if dataset else 1000) * args.epochs
    print(f"training: {args.epochs} epochs, lr={args.lr}")
    print()
    print("training...")
    print("─────────────────────────────────────────────────────")

    t0 = time.time()
    best_loss = 99.0
    global_step = 0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        if dataset:
            indices = list(range(len(dataset)))
            random.seed(42 + epoch)
            random.shuffle(indices)
        else:
            indices = list(range(1000))

        step_in_epoch = 0
        for idx in indices:
            if epoch == start_epoch and step_in_epoch < start_step:
                step_in_epoch += 1
                global_step += 1
                continue

            lr = get_lr(global_step, total_steps, args.lr)

            if dataset:
                conversations, image_bytes_list = dataset[idx]
                token_ids, target_ids = build_training_pair(
                    conversations,
                    vlm_config.image_special_token,
                    vlm_config.image_token_len,
                    vlm_config.max_seq_len,
                    tokenizer
                )

                # Forward with image
                loss_idx, loss_val = model.forward_train_vlm(
                    token_ids, target_ids, image_bytes_list
                )
            else:
                # Synthetic data for testing
                token_ids = [random.randint(0, 99) for _ in range(vlm_config.max_seq_len)]
                target_ids = token_ids[1:] + [0]
                loss_idx, loss_val = model.forward_train(token_ids, target_ids)

            if loss_val < best_loss:
                best_loss = loss_val

            model.backward_step(loss_idx, loss_val, lr)

            step_in_epoch += 1
            global_step += 1

            if global_step % args.log_every == 0 or global_step == 1:
                elapsed = time.time() - t0
                print(f"  epoch {epoch+1}/{args.epochs} step {step_in_epoch:5d} | "
                      f"loss {loss_val:.4f} | best {best_loss:.4f} | "
                      f"lr {lr:.2e} | {elapsed:.1f}s")

            if global_step % args.save_every == 0:
                save_path = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}.bin'
                model.save_weights(save_path)
                with open(save_path + '.meta', 'w') as f:
                    f.write(f"{step_in_epoch}\n{epoch}\n{best_loss}\n"
                            f"{vlm_config.hidden_size}\n{vlm_config.num_hidden_layers}\n"
                            f"{vlm_config.max_seq_len}\n")
                print(f"  ──── saved checkpoint (step {global_step})")

    elapsed = time.time() - t0
    print("─────────────────────────────────────────────────────")
    print(f"  best loss: {best_loss:.4f}")
    print(f"  time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save final
    save_path = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}.bin'
    model.save_weights(save_path)
    with open(save_path + '.meta', 'w') as f:
        f.write(f"{step_in_epoch}\n{epoch}\n{best_loss}\n"
                f"{vlm_config.hidden_size}\n{vlm_config.num_hidden_layers}\n"
                f"{vlm_config.max_seq_len}\n")

    print()
    print("════════════════════════════════════════════════════════")
    print(f"  MiniMind-V {n_params:,} params. No PyTorch.")
    print("════════════════════════════════════════════════════════")


if __name__ == '__main__':
    main()
