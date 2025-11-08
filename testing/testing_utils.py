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