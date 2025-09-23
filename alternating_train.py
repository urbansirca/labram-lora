from meta_train import get_meta_engine
from train import get_engine
import json
import yaml

from models import DeepConvNet, EEGNet, load_labram

def alternating_train(
    engine,              # Engine
    meta_engine,         # MetaEngine
    total_epochs: int,   # total "cycles": 1 normal + 1 meta
    meta_iters_per_meta_epoch: int,  # how many meta-iterations constitute 1 "meta epoch"
    validate_meta_every: int = 0,    # 0 = only at meta epoch end
):
    # sanity: ensure same model object
    assert engine.model is meta_engine.model, "Engine and MetaEngine must share the same model instance"
    
    # assert val and train subjects are the same
    # assert engine.S_val == meta_engine.S_val, "Engine and MetaEngine must have the same validation subjects"
    # assert engine.S_train == meta_engine.S_train, "Engine and MetaEngine must have the same training subjects"

    for epoch in range(1, total_epochs + 1):
        # ---------- 1) supervised epoch (normal loss) ----------
        engine.metrics.epoch = epoch
        engine.train_epoch(engine.training_set)
        

        # ---------- 2) meta epoch (meta loss) ----------
        # one "meta epoch" = N meta-iterations
        for i in range(1, meta_iters_per_meta_epoch + 1):
            T = min(meta_engine.T, len(meta_engine.S_train))
            subjects_batch = meta_engine.rng.sample(meta_engine.S_train, k=T)
            meta_engine.meta_step(subjects_batch)

            if meta_engine.scheduler is not None:
                meta_engine.scheduler.step()
        

        engine.validate_epoch()
        engine.log_metrics()
        engine.save_regular_checkpoint()

        meta_engine.validate_epoch()
        meta_engine.log_metrics()
        meta_engine.save_regular_checkpoint()



if __name__ == "__main__":

    with open("alternating_hyperparameters.yaml", "r") as f:
        common_config = yaml.safe_load(f)

    shared_config = common_config["shared"]
    config = common_config["train"]
    meta_config = common_config["meta"]


    
    model = load_labram(
        lora=True,
        peft_config=shared_config["peft"],
    )


    experiment_name = "alternating_train"
    engine = get_engine(config, with_tester=False, experiment_name=experiment_name, model=model, model_str="labram", model_hyperparameters=shared_config["labram_hp"])
    meta_engine = get_meta_engine(meta_config, with_tester=False, experiment_name=experiment_name, model=model, model_str="labram", model_hyperparameters=shared_config["labram_hp"])

    alternating_train(engine, meta_engine, total_epochs=50, meta_iters_per_meta_epoch=50, validate_meta_every=1)




