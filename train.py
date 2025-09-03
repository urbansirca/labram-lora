import random
import wandb
import yaml
import logging
from datetime import datetime

from experiment import Experiment
from labram import load_labram
from utils import get_optimizer_scheduler

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # This makes logs appear in terminal
)
logger = logging.getLogger(__name__)


with open("hyperparameters.yaml", "r") as f:
    config = yaml.safe_load(f)


experiment_name = "_".join(
    [config["experiment"]["model"], datetime.now().strftime("%H%M%S")]
)

logger.info(f"Experiment name: {experiment_name}")
logger.info(f"Experiment config: {config['experiment']}")

if config["experiment"]["model"] == "labram":
    hyperparameters = config["labram"]
    logger.info(f"HYPERPARAMETERS for labram: {hyperparameters}")
    model = load_labram()
elif config["experiment"]["model"] == "deepconvnet":
    raise ValueError("DeepConvNet is not implemented yet")
else:
    raise ValueError("Invalid model")


N_EPOCHS = config["experiment"]["epochs"]
DEVICE = config["experiment"]["device"]
BATCH_SIZE = hyperparameters["batch_size"]
DEVICE = config["experiment"]["device"]
SEED = config["experiment"]["seed"]
SHUFFLED_SUBJECTS = config["experiment"]["shuffled_subjects"]
LOMSO_FOLDS = config["experiment"]["LOMSO_folds"]
META = config["experiment"]["meta"]

optimizer, scheduler = get_optimizer_scheduler(
    config["experiment"]["optimizer"], config["experiment"]["scheduler"]
)

# Data
training_set = None
validation_set = None
test_set = None


experiment = Experiment(
    model=model,
    config=config,  # for wandb
    hyperparameters=hyperparameters,
    experiment_name=experiment_name,
    n_epochs=N_EPOCHS,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    training_set=training_set,
    validation_set=validation_set,
    test_set=test_set,
)
