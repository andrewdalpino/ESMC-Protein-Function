import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.backends.mps import is_available as mps_is_available
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_

from metrics import F1Score

from torch.utils.tensorboard import SummaryWriter

from esm.tokenization import EsmSequenceTokenizer

import obonet

from src.esmc_protein_function.model import ESMCProteinFunction
from data import AmiGOBoost, LengthBucketBatchSampler, SortedLengthBatchSampler
from loss import AdaptiveAspectWeighting

from tqdm import tqdm

AVAILABLE_BASE_MODELS = ESMCProteinFunction.ESM_PRETRAINED_CONFIGS.keys()

ONE_THIRD = 1 / 3


def main():
    parser = ArgumentParser(
        description="Fine-tune an ESMC model for Gene Ontology (GO) term prediction."
    )

    parser.add_argument(
        "--base_model",
        default="esmc_300m",
        choices=AVAILABLE_BASE_MODELS,
    )

    parser.add_argument("--num_dataset_processes", default=1, type=int)
    parser.add_argument("--go_db_path", default="./dataset/go-basic.obo", type=str)
    parser.add_argument("--min_sequence_length", default=1, type=int)
    parser.add_argument("--max_sequence_length", default=2048, type=int)
    parser.add_argument("--unfreeze_last_k_layers", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--aspect_learning_rate", default=1e-3, type=float)
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_length_buckets", default=100, type=int)
    parser.add_argument("--mf_aspect_ratio", default=ONE_THIRD, type=float)
    parser.add_argument("--bp_aspect_ratio", default=ONE_THIRD, type=float)
    parser.add_argument("--cc_aspect_ratio", default=ONE_THIRD, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--max_steps_per_epoch", default=2048, type=int)
    parser.add_argument("--num_mf_pool_heads", default=8, type=int)
    parser.add_argument("--num_bp_pool_heads", default=16, type=int)
    parser.add_argument("--num_cc_pool_heads", default=4, type=int)
    parser.add_argument("--num_mf_layers", default=2, type=int)
    parser.add_argument("--num_bp_layers", default=3, type=int)
    parser.add_argument("--num_cc_layers", default=1, type=int)
    parser.add_argument("--use_flash_attention", default=True, type=bool)
    parser.add_argument("--quantization_aware_training", action="store_true")
    parser.add_argument("--quant_group_size", default=64, type=int)
    parser.add_argument("--eval_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=5, type=int)

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.num_epochs < 1:
        raise ValueError(f"Must train for at least 1 epoch, {args.num_epochs} given.")

    if args.eval_interval < 1:
        raise ValueError(
            f"Eval interval must be greater than 0, {args.eval_interval} given."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            f"Checkpoint interval must be greater than 0, {args.checkpoint_interval} given."
        )

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    if "mps" in args.device and not mps_is_available():
        raise RuntimeError("MPS is not available.")

    torch.set_float32_matmul_precision("high")

    dtype = (
        torch.bfloat16
        if "cuda" in args.device and is_bf16_supported()
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    tokenizer = EsmSequenceTokenizer()

    graph = obonet.read_obo(args.go_db_path)

    new_dataset = partial(
        AmiGOBoost,
        graph=graph,
        tokenizer=tokenizer,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
    )

    mf_train = new_dataset(subset="mf", split="train")
    bp_train = new_dataset(subset="bp", split="train")
    cc_train = new_dataset(subset="cc", split="train")

    mf_test = new_dataset(subset="mf", split="test")
    bp_test = new_dataset(subset="bp", split="test")
    cc_test = new_dataset(subset="cc", split="test")

    # Make sure test and train sets have the same label indices to GO term mapping.
    mf_test.go_ids_to_label_indices = mf_train.go_ids_to_label_indices
    bp_test.go_ids_to_label_indices = bp_train.go_ids_to_label_indices
    cc_test.go_ids_to_label_indices = cc_train.go_ids_to_label_indices

    new_dataloader = partial(
        DataLoader,
        collate_fn=mf_train.collate_pad_right,
        pin_memory="cuda" in args.device,
        num_workers=args.num_dataset_processes,
    )

    new_sampler = partial(
        LengthBucketBatchSampler,
        batch_size=args.batch_size,
        num_buckets=args.num_length_buckets,
    )

    mf_train_loader = new_dataloader(mf_train, batch_sampler=new_sampler(mf_train))
    bp_train_loader = new_dataloader(bp_train, batch_sampler=new_sampler(bp_train))
    cc_train_loader = new_dataloader(cc_train, batch_sampler=new_sampler(cc_train))

    new_sampler = partial(
        SortedLengthBatchSampler,
        batch_size=args.batch_size,
    )

    mf_test_loader = new_dataloader(mf_test, batch_sampler=new_sampler(mf_test))
    bp_test_loader = new_dataloader(bp_test, batch_sampler=new_sampler(bp_test))
    cc_test_loader = new_dataloader(cc_test, batch_sampler=new_sampler(cc_test))

    model_args = {
        "model_name": args.base_model,
        "num_mf_pool_heads": args.num_mf_pool_heads,
        "num_bp_pool_heads": args.num_bp_pool_heads,
        "num_cc_pool_heads": args.num_cc_pool_heads,
        "num_mf_layers": args.num_mf_layers,
        "num_bp_layers": args.num_bp_layers,
        "num_cc_layers": args.num_cc_layers,
        "index_to_mf_term": mf_train.label_indices_to_go_ids,
        "index_to_bp_term": bp_train.label_indices_to_go_ids,
        "index_to_cc_term": cc_train.label_indices_to_go_ids,
        "use_flash_attention": args.use_flash_attention,
    }

    model = ESMCProteinFunction.from_esm_pretrained(**model_args)

    model.freeze_base()

    model.unfreeze_last_k_encoder_layers(args.unfreeze_last_k_layers)

    if args.quantization_aware_training:
        model.add_fake_quantized_tensors(args.quant_group_size)

    print(f"Number of parameters: {model.num_params:,}")
    print(f"Number of trainable parameters: {model.num_trainable_parameters:,}")

    model = model.to(args.device)

    loss_function = BCEWithLogitsLoss()

    aspect_weight = AdaptiveAspectWeighting({"MF", "BP", "CC"}, 0.1).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    aspect_optimizer = SGD(aspect_weight.parameters(), lr=args.aspect_learning_rate)

    f1_metric = F1Score().to(args.device)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=False
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        aspect_weight.load_state_dict(checkpoint["aspect_weight"])
        aspect_optimizer.load_state_dict(checkpoint["aspect_optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    model.train()

    train_paths = [
        ("MF", model.forward_mf, iter(mf_train_loader)),
        ("BP", model.forward_bp, iter(bp_train_loader)),
        ("CC", model.forward_cc, iter(cc_train_loader)),
    ]

    aspect_ratios = [
        args.mf_aspect_ratio,
        args.bp_aspect_ratio,
        args.cc_aspect_ratio,
    ]

    total_weight = sum(aspect_ratios)

    aspect_probs = [r / total_weight for r in aspect_ratios]

    sample_aspect = lambda: random.choices(train_paths, aspect_probs, k=1)[0]

    test_paths = [
        ("MF", model.predict_mf, mf_test_loader),
        ("BP", model.predict_bp, bp_test_loader),
        ("CC", model.predict_cc, cc_test_loader),
    ]

    new_progress_bar = partial(
        tqdm,
        total=args.max_steps_per_epoch,
        leave=False,
    )

    print("Fine-tuning ...")

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        step, total_batches, total_steps = 1, 0, 0

        progress = new_progress_bar(desc=f"Epoch {epoch}")

        optimizer.zero_grad()
        aspect_optimizer.zero_grad()

        while step <= args.max_steps_per_epoch:
            aspect, forward, dataloader = sample_aspect()

            x, y = next(dataloader)

            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with amp_context:
                y_pred = forward(x)

                bce_loss = loss_function.forward(y_pred, y)

                weighted_loss = aspect_weight.forward(bce_loss, aspect)

                scaled_loss = weighted_loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            total_cross_entropy += bce_loss.item()
            total_batches += 1

            if step % args.gradient_accumulation_steps == 0:
                norm = clip_grad_norm_(model.parameters(), args.max_gradient_norm)
                _ = clip_grad_norm_(aspect_weight.parameters(), args.max_gradient_norm)

                optimizer.step()
                aspect_optimizer.step()

                optimizer.zero_grad()
                aspect_optimizer.zero_grad()

                total_gradient_norm += norm.item()
                total_steps += 1

            progress.update(1)

            step += 1

        progress.close()

        average_cross_entropy = total_cross_entropy / total_batches
        average_gradient_norm = total_gradient_norm / total_steps

        mf_weight, bp_weight, cc_weight = aspect_weight.weights

        logger.add_scalar("Cross Entropy", average_cross_entropy, epoch)
        logger.add_scalar("Gradient Norm", average_gradient_norm, epoch)
        logger.add_scalar("MF Weight", mf_weight, epoch)
        logger.add_scalar("BP Weight", bp_weight, epoch)
        logger.add_scalar("CC Weight", cc_weight, epoch)

        print(
            f"Epoch {epoch}:",
            f"Cross Entropy: {average_cross_entropy:.5f},",
            f"Gradient Norm: {average_gradient_norm:.5f}",
        )

        if epoch % args.eval_interval == 0:
            model.eval()

            for aspect, forward, dataloader in test_paths:
                for x, y in tqdm(dataloader, desc=f"Testing {aspect}", leave=False):
                    x = x.to(args.device, non_blocking=True)
                    y = y.to(args.device, non_blocking=True)

                    y_prob = forward(x)

                    f1_metric.update(y_prob, y)

                f1_score, precision, recall = f1_metric.compute()

                logger.add_scalar(f"{aspect} F1 Score", f1_score, epoch)
                logger.add_scalar(f"{aspect} Precision", precision, epoch)
                logger.add_scalar(f"{aspect} Recall", recall, epoch)

                print(
                    f"{aspect} F1 Score: {f1_score:.4f}, "
                    f"{aspect} Precision: {precision:.4f}, "
                    f"{aspect} Recall: {recall:.4f}"
                )

                f1_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_args": model_args,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aspect_weight": aspect_weight.state_dict(),
                "aspect_optimizer": aspect_optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")

    print("Done!")


if __name__ == "__main__":
    main()
