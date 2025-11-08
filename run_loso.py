from testing.LOSO import run_loso
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run LOSO evaluation")
    parser.add_argument(
        "--model",
        type=str,
        choices=["labram-lora", "labram-partialft", "deepconvnet"],
        required=True,
        help="Model type to evaluate (labram-lora, labram-partialft, or deepconvnet)",
    )
    args = parser.parse_args()
    model = args.model

    # Load base configuration
    cfg_path = Path(f"hyperparameters/hyperparameters-{model}.yaml")
   
    run_loso(cfg_path)

if __name__ == "__main__":
    main()