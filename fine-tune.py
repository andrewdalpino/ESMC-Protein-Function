import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.backends.mps import is_available as mps_is_available
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_

from torchmetrics.classification import BinaryPrecision, BinaryRecall

from torch.utils.tensorboard import SummaryWriter

from esm.tokenization import EsmSequenceTokenizer

import obonet

from src.esmc_function_classifier.model import EsmcGoTermClassifier
from data import AmiGOBoost

from tqdm import tqdm

AVAILABLE_BASE_MODELS = EsmcGoTermClassifier.ESM_PRETRAINED_CONFIGS.keys()


def main():
    parser = ArgumentParser(
        description="Fine-tune an ESMC model for gene ontology (GO) term prediction."
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
    parser.add_argument("--quantization_aware_training", action="store_true")
    parser.add_argument("--quant_group_size", default=192, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--max_steps_per_epoch", default=1024, type=int)
    parser.add_argument("--use_flash_attention", default=True, type=bool)
    parser.add_argument("--eval_interval", default=2, type=int)
    parser.add_argument("--checkpoint_interval", default=2, type=int)
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

    mf_test.num_classes = mf_train.num_classes
    bp_test.num_classes = bp_train.num_classes
    cc_test.num_classes = cc_train.num_classes

    new_dataloader = partial(
        DataLoader,
        batch_size=args.batch_size,
        collate_fn=mf_train.collate_pad_right,
        pin_memory="cuda" in args.device,
        num_workers=args.num_dataset_processes,
    )

    mf_train_loader = new_dataloader(mf_train, shuffle=True)
    bp_train_loader = new_dataloader(bp_train, shuffle=True)
    cc_train_loader = new_dataloader(cc_train, shuffle=True)

    mf_test_loader = new_dataloader(mf_test)
    bp_test_loader = new_dataloader(bp_test)
    cc_test_loader = new_dataloader(cc_test)

    model_args = {
        "model_name": args.base_model,
        "indexToMfGoTerm": mf_train.label_indices_to_go_ids,
        "indexToBpGoTerm": bp_train.label_indices_to_go_ids,
        "indexToCcGoTerm": cc_train.label_indices_to_go_ids,
        "use_flash_attention": args.use_flash_attention,
    }

    model = EsmcGoTermClassifier.from_esm_pretrained(**model_args)

    model.freeze_base()

    model.unfreeze_last_k_encoder_layers(args.unfreeze_last_k_layers)

    if args.quantization_aware_training:
        model.add_fake_quantized_tensors(args.quant_group_size)

    print(f"Number of parameters: {model.num_params:,}")
    print(f"Number of trainable parameters: {model.num_trainable_parameters:,}")

    model = model.to(args.device)

    loss_function = BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    precision_metric = BinaryPrecision().to(args.device)
    recall_metric = BinaryRecall().to(args.device)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=False
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    model.train()

    print("Fine-tuning ...")

    train_loaders = [
        ("mf", iter(mf_train_loader)),
        ("bp", iter(bp_train_loader)),
        ("cc", iter(cc_train_loader)),
    ]

    test_loaders = [
        ("mf", iter(mf_test_loader)),
        ("bp", iter(bp_test_loader)),
        ("cc", iter(cc_test_loader)),
    ]

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0
        step = 0

        progress = tqdm(
            total=args.max_steps_per_epoch, desc=f"Epoch {epoch}", leave=False
        )

        while step < args.max_steps_per_epoch:
            aspect, dataloader = random.choice(train_loaders)

            x, y = next(dataloader)

            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            match aspect:
                case "mf":
                    forward_path = model.forward_mf
                case "bp":
                    forward_path = model.forward_bp
                case "cc":
                    forward_path = model.forward_cc

            with amp_context:
                y_pred = forward_path(x)

                loss = loss_function(y_pred, y)

                scaled_loss = loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            total_cross_entropy += loss.item()
            total_batches += 1

            if step % args.gradient_accumulation_steps == 0:
                norm = clip_grad_norm_(model.parameters(), args.max_gradient_norm)

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                total_gradient_norm += norm.item()
                total_steps += 1

            progress.update(1)

            step += 1

        average_cross_entropy = total_cross_entropy / total_batches
        average_gradient_norm = total_gradient_norm / total_steps

        logger.add_scalar("Cross Entropy", average_cross_entropy, epoch)
        logger.add_scalar("Gradient Norm", average_gradient_norm, epoch)

        print(
            f"Epoch {epoch}:",
            f"Cross Entropy: {average_cross_entropy:.5f},",
            f"Gradient Norm: {average_gradient_norm:.5f}",
        )

        if epoch % args.eval_interval == 0:
            model.eval()

            for aspect, dataloader in test_loaders:
                match aspect:
                    case "mf":
                        forward_path = model.forward_mf
                    case "bp":
                        forward_path = model.forward_bp
                    case "cc":
                        forward_path = model.forward_cc

                for x, y in tqdm(dataloader, desc=f"Testing {aspect}", leave=False):
                    x = x.to(args.device, non_blocking=True)
                    y = y.to(args.device, non_blocking=True)

                    with torch.no_grad(), amp_context:
                        y_pred = forward_path(x)

                        y_prob = torch.sigmoid(y_pred)

                    precision_metric.update(y_prob, y)
                    recall_metric.update(y_prob, y)

            precision = precision_metric.compute()
            recall = recall_metric.compute()

            f1_score = (2 * precision * recall) / (precision + recall)

            logger.add_scalar("F1 Score", f1_score, epoch)
            logger.add_scalar("Precision", precision, epoch)
            logger.add_scalar("Recall", recall, epoch)

            print(
                f"F1: {f1_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
            )

            precision_metric.reset()
            recall_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_args": model_args,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")

    print("Done!")


if __name__ == "__main__":
    main()
