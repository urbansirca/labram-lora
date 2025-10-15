from testing.run_lomso import run_lomso
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run LOMSO training/testing pipeline.")
    parser.add_argument(
        "--with-meta",
        action="store_true",
        help="Use the meta-learning engine instead of the standard one",
    )
    parser.add_argument(
        "--run-loso",
        action="store_true",
        help="Use leave-one-subject-out (LOSO) evaluation instead of LOMSO.",
    )
    args = parser.parse_args()

    if args.with_meta:
        run_lomso("hyperparameters/meta_hyperparameters.yaml", with_meta=True, run_loso=args.run_loso)
    else:
        run_lomso("hyperparameters/hyperparameters.yaml", with_meta=False, run_loso=args.run_loso)

if __name__ == "__main__":
    main()