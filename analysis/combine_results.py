import pandas as pd
import pathlib

root_dir = pathlib.Path("ng_testing/raw_dataset")
models = [""]


# loop over models and folds, collect results
all_results = []
for model in models:
    model_dir = pathlib.Path(root_dir) / model
    for experiment_dir in model_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
        results_file = experiment_dir / "repetition_results.csv"
        print(results_file)
        if not results_file.exists():
            print(f"Warning: {results_file} does not exist, skipping.")
            continue
        df = pd.read_csv(results_file)
        df["model"] = "deepconvnet_alt_raw"
        df["experiment"] = experiment_dir.name
        all_results.append(df)
        
# concatenate all results into one dataframe and save
if all_results:
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(pathlib.Path(root_dir) / "all_results.csv", index=False)
    print(f"Saved all results to {pathlib.Path(root_dir) / 'all_results.csv'}")