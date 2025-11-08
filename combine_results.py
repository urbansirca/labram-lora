import pandas as pd
import pathlib

root_dir = pathlib.Path("LOSO_TEST")
models = ["labram-lora","labram-partialft","deepconvnet"]


# loop over models and folds, collect results
combined_results = []
for model in models:
    model_dir = pathlib.Path(root_dir) / model
    for experiment_dir in model_dir.iterdir():
        if not experiment_dir.is_dir():
            raise StopIteration
        results_file = experiment_dir / "TEST_RESULTS/repetition_results.csv"
        if not results_file.exists():
            raise StopIteration
        df = pd.read_csv(results_file)
        df["model"] = f"{model}"
        df["experiment"] = experiment_dir.name
        combined_results.append(df)
        
if combined_results:
    combined_results_df = pd.concat(combined_results, ignore_index=True)
    combined_results_df.to_csv(pathlib.Path(root_dir) / "COMBINED_RESULTS.csv", index=False)
    print(f"Saved all results to {pathlib.Path(root_dir) / 'COMBINED_RESULTS.csv'}")