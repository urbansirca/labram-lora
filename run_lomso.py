from testing.run_lomso import run_lomso
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run LOMSO training/testing pipeline.")
    parser.add_argument(
        "--run-loso",
        action="store_true",
        help="Use leave-one-subject-out (LOSO) evaluation instead of LOMSO.",
    )
    args = parser.parse_args()

    run_lomso("hyperparameters/hyperparameters.yaml", run_loso=args.run_loso)

if __name__ == "__main__":
    main()