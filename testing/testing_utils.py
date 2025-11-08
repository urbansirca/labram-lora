from pathlib import Path
import re

def build_subject_list(config):
    data_cfg = config.get("data")
    exp_cfg = config.get("experiment")
    subjects = data_cfg.get("subjects")
    if subjects is None:
        n = exp_cfg.get("n_subjects")
        if n is None:
            raise ValueError(
                "No subjects list in config and experiment.n_subjects not set"
            )
        subjects = list(range(1, n + 1))
    return list(subjects)

def load_best_checkpoint(folder: str):
    """
    Given a checkpoint folder path, finds the checkpoint with the highest accuracy.

    The filenames are expected to contain patterns like 'acc0.705'.
    Returns the path to the best checkpoint and its accuracy.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Regex to capture accuracy from filenames like checkpoint_i600_acc0.705
    acc_pattern = re.compile(r"acc([0-9]*\.?[0-9]+)")

    best_acc = -1.0
    best_ckpt_path = None

    for file in folder.iterdir():
        if file.is_dir():  # skip dirs
            continue
        match = acc_pattern.search(file.name)
        if match:
            acc = float(match.group(1))
            if acc > best_acc:
                best_acc = acc
                # Prefer checkpoint folder if it exists, otherwise the file
                base_name = file.stem.split("_training_metadata")[0]
                ckpt_folder = folder / base_name
                best_ckpt_path = ckpt_folder if ckpt_folder.exists() else file

    if best_ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint files with 'acc' pattern found in {folder}")

    print(f"Best checkpoint: {best_ckpt_path.name} (acc={best_acc:.3f})")
    return best_ckpt_path, best_acc

def get_checkpoint_file(root_dir, model_name,leave_out_subjs, type="best", head_only=False):
    """
    gets the checkpoint file for a given model and leave_out subjects for LOMSO. 
    type can be best or final
    """
    
    model_dir = Path(root_dir) / model_name
    
    # for labram head_only, the model dir is labram_head_only
    if model_name == "labram" and head_only:
        model_dir = Path(root_dir) / "labram_head_only"
    
    print(f"Looking for model directory {model_dir}")
    if not model_dir.exists():
        raise ValueError(f"Model directory {model_dir} does not exist")
    ckpt_file = None
    # find the experiment dir with the matching leave_out subjects
    for experiment_dir in model_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
        # check if leave_out matches
        check = f"test{leave_out_subjs}"
        
        if check == experiment_dir.name.split('_')[-1]:
            print(f"Found experiment directory: {experiment_dir}")
            if model_name == "deepconvnet":
                if type == "best":
                    ckpt_file = experiment_dir / "best_val_checkpoint.pt"
                    # if file not found, fallback to load_best_checkpoint
                    if not ckpt_file.exists():
                        try:
                            ckpt_file, _ = load_best_checkpoint(experiment_dir)
                        except Exception as e:
                            raise ValueError(f"No best_val_checkpoint.pt or fallback found in {experiment_dir}: {e}")
                elif type == "final":
                    final_ckpt_files = list(experiment_dir.glob("FINAL*.pt"))
                    if not final_ckpt_files:
                        raise ValueError(f"No final_checkpoint*.pt file found in {experiment_dir}")
                    if len(final_ckpt_files) > 1:
                        raise ValueError(f"Multiple final_checkpoint*.pt files found in {experiment_dir}")
                    ckpt_file = final_ckpt_files[0]
            elif model_name == "labram":
                if type == "best":
                    ckpt_file = experiment_dir / "best_val_checkpoint"
                    if not ckpt_file.exists():
                        try:
                            ckpt_file, _ = load_best_checkpoint(experiment_dir)
                        except Exception as e:
                            raise ValueError(f"No best_val_checkpoint directory or fallback found in {experiment_dir}: {e}")
                elif type == "final":
                    # find the subfolder that starts with final_checkpoint (there should be only one)
                    final_ckpt_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("FINAL")]
                    if not final_ckpt_dirs:
                        raise ValueError(f"No final_checkpoint* directory found in {experiment_dir}")
                    if len(final_ckpt_dirs) > 1:
                        raise ValueError(f"Multiple final_checkpoint* directories found in {experiment_dir}")
                    ckpt_file = final_ckpt_dirs[0]
            else:
                raise ValueError(f"Unknown model name {model_name}")
            
    if ckpt_file is None:
        raise ValueError(f"No matching experiment directory found for leave_out={leave_out_subjs} in {model_dir}")

    print(f"Found checkpoint file {ckpt_file} for model {model_name} leave_out {leave_out_subjs} type {type}")
    return ckpt_file