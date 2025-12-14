import argparse
import importlib.util
from pathlib import Path
import torch
import tiktoken


def load_training_module():
    """
    Dynamically import pico-llm.py so we can reuse the model classes and generate_text utility.
    """
    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / "pico-llm.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot locate training script at {module_path}")

    spec = importlib.util.spec_from_file_location("pico_llm_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(module, checkpoint, device):
    """
    Recreate the correct model architecture from the metadata stored in the checkpoint.
    """
    config = checkpoint.get("config", {})
    model_type = checkpoint.get("model_type") or config.get("model_type")
    if not model_type:
        raise ValueError("Checkpoint missing 'model_type' metadata. Cannot rebuild model.")

    if model_type == "kgram_mlp_seq":
        model = module.KGramMLPSeqModel(
            vocab_size=config["vocab_size"],
            k=config["kgram_k"],
            embed_size=config["embed_size"],
            num_inner_layers=config.get("num_inner_layers", 1),
            chunk_size=config.get("chunk_size", 1),
        )
    elif model_type == "lstm_seq":
        model = module.LSTMSeqModel(
            vocab_size=config["vocab_size"],
            embed_size=config["embed_size"],
            hidden_size=config.get("hidden_size", config["embed_size"]),
        )
    elif model_type == "transformer":
        model = module.TransformerModel(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_blocks=config["n_blocks"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            dropout=config.get("dropout", 0.0),
            max_seq_len=config.get("max_seq_len", 1024),
            positional_embedding=config.get("positional_embedding", "learned"),
            rope_base=config.get("rope_base", 10000.0),
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' in checkpoint.")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, model_type


def parse_args():
    parser = argparse.ArgumentParser(description="Use a saved pico-llm checkpoint to generate text.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the .pt checkpoint saved by pico-llm.py.")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt text to start generation.")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="How many new tokens to sample from the model.")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Optional nucleus sampling threshold. Leave empty for greedy decoding.")
    parser.add_argument("--device_id", type=str, default="cpu",
                        help="Device to load the model onto, e.g., 'cpu' or 'cuda:0'.")
    return parser.parse_args()


def main():
    args = parse_args()
    requested_device = args.device_id
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device}' but CUDA is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    module = load_training_module()
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model, model_type = build_model(module, checkpoint, device)

    enc = tiktoken.get_encoding("gpt2")
    text, annotated = module.generate_text(
        model,
        enc,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
        top_p=args.top_p,
    )

    print(f"[{model_type}] Prompt: {args.prompt}")
    print(text)
    print("\n--- Annotated ---")
    print(annotated)


if __name__ == "__main__":
    main()
