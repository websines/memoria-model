"""CLI entry point: python -m memoria [command]"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Memoria Neural Architecture")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a Memoria model")
    train_parser.add_argument("--config", choices=["small", "medium", "large", "full", "qwen", "lfm2"], default="small")
    # Sentinel default: None means "user didn't set it". We resolve to 5000 or a
    # huge cap below depending on whether --time-budget was given.
    train_parser.add_argument("--max-steps", type=int, default=None)
    train_parser.add_argument("--time-budget", type=float, default=None, help="Max training time in seconds")
    train_parser.add_argument("--checkpoint-dir", default="checkpoints")
    train_parser.add_argument("--no-wandb", action="store_true")
    train_parser.add_argument("--no-hub", action="store_true")
    train_parser.add_argument("--resume", type=str, default=None)

    # Test
    subparsers.add_parser("test", help="Run tests")

    # Info
    info_parser = subparsers.add_parser("info", help="Show model info for a config")
    info_parser.add_argument("--config", choices=["small", "medium", "large", "full", "qwen", "lfm2"], default="small")

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("checkpoint", help="Path to checkpoint")
    eval_parser.add_argument("--config", choices=["small", "medium", "large", "full", "qwen", "lfm2"], default="small")
    eval_parser.add_argument("--suite", choices=["all", "perplexity", "belief", "causal", "telos", "improvement", "crossover"], default="all")

    args = parser.parse_args()

    if args.command == "train":
        from .model.config import small_config, medium_config, large_config, full_config, qwen_config, lfm2_config
        from .training.train import train

        configs = {"small": small_config, "medium": medium_config, "large": large_config, "full": full_config, "qwen": qwen_config, "lfm2": lfm2_config}
        config = configs[args.config]()

        # Resolve max_steps default: if --time-budget is set and --max-steps is
        # not explicitly set, use a huge cap so the time budget becomes the true
        # stop condition. Otherwise, fall back to the historical 5000-step default.
        if args.max_steps is None:
            resolved_max_steps = 10**9 if args.time_budget is not None else 5000
        else:
            resolved_max_steps = args.max_steps

        budget_note = f", time_budget={args.time_budget}s" if args.time_budget else ""
        print(f"Training with {args.config} config, max_steps={resolved_max_steps}{budget_note}")
        train(
            config=config,
            max_steps=resolved_max_steps,
            time_budget=args.time_budget,
            checkpoint_dir=args.checkpoint_dir,
            log_to_wandb=not args.no_wandb,
            push_to_hub=not args.no_hub,
            resume_from=args.resume,
        )

    elif args.command == "info":
        from .model.config import small_config, medium_config, large_config, full_config, qwen_config, lfm2_config
        from .model.memoria_model import MemoriaModel
        from .model.pretrained_model import PretrainedMemoriaModel

        configs = {"small": small_config, "medium": medium_config, "large": large_config, "full": full_config, "qwen": qwen_config, "lfm2": lfm2_config}
        config = configs[args.config]()
        if config.backbone == "pretrained":
            model = PretrainedMemoriaModel(config)
        else:
            model = MemoriaModel(config)
        print(model.summary())

    elif args.command == "test":
        import pytest
        sys.exit(pytest.main(["-v", "tests/"]))

    elif args.command == "eval":
        _run_eval(args)

    else:
        parser.print_help()


def _run_eval(args):
    import torch
    from .model.config import small_config, medium_config, large_config, full_config, qwen_config, lfm2_config
    from .model.memoria_model import MemoriaModel
    from .model.pretrained_model import PretrainedMemoriaModel

    configs = {"small": small_config, "medium": medium_config, "large": large_config, "full": full_config, "qwen": qwen_config, "lfm2": lfm2_config}
    config = configs[args.config]()

    print(f"Loading checkpoint: {args.checkpoint}")
    if config.backbone == "pretrained":
        model = PretrainedMemoriaModel(config)
    else:
        model = MemoriaModel(config)
    ckpt = torch.load(args.checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.state.load_state_cognitive(ckpt['cognitive_state'])
    if torch.cuda.is_available():
        model = model.cuda()

    suites = {
        "perplexity": _eval_perplexity,
        "belief": _eval_belief,
        "causal": _eval_causal,
        "telos": _eval_telos,
        "improvement": _eval_improvement,
        "crossover": _eval_crossover,
    }

    if args.suite == "all":
        for name, fn in suites.items():
            print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
            fn(model)
    else:
        suites[args.suite](model)


def _eval_perplexity(model):
    from .eval.perplexity import evaluate_perplexity
    print(evaluate_perplexity(model, num_batches=50))

def _eval_belief(model):
    from .eval.belief_tracking import evaluate_belief_tracking
    print(evaluate_belief_tracking(model))

def _eval_causal(model):
    from .eval.causal import evaluate_causal_reasoning
    print(evaluate_causal_reasoning(model))

def _eval_telos(model):
    from .eval.telos_demo import evaluate_telos
    print(evaluate_telos(model))

def _eval_improvement(model):
    from .eval.improvement import evaluate_improvement_curve, save_improvement_results
    results = evaluate_improvement_curve(model)
    save_improvement_results(results, "results/improvement_curve.json")
    print(f"Accuracies: {results['accuracies']}")

def _eval_crossover(model):
    from .eval.crossover import evaluate_crossover
    results = evaluate_crossover(model, output_path="results/crossover.json")
    print(f"Crossover point: {results['crossover_point']}")


if __name__ == "__main__":
    main()
