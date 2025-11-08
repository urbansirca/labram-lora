from torcheeg.models.transformer.labram import LaBraM
import torch
from collections import OrderedDict
import pathlib
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
import yaml
import logging
from peft import PeftModel
from pathlib import Path
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_labram(lora=True, peft_config=None):
    # Load checkpoint first
    MODEL_PATH = (
        pathlib.Path(__file__).parent.parent
        / "weights"
        / "pretrained-models"
        / "labram-base.pth"
    )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    processed_state = {}
    for key, value in checkpoint["model"].items():
        if key.startswith("student."):
            new_key = key.replace("student.", "")  # Remove 'student.' prefix
            # Map student.norm to fc_norm (the actual LayerNorm layer)
            if new_key.startswith("norm."):
                new_key = "fc_norm" + new_key[4:]  # norm.weight -> fc_norm.weight
            processed_state[new_key] = value
        else:
            processed_state[key] = value

    # Create model and load processed weights
    if not lora:
        model = LaBraM.base_patch200_200(
            num_classes=2,
        )  # TODO: not sure if the head should be initialized here or added on top of the model
    else:
        model = LaBraM.base_patch200_200(
            num_classes=2,
        )  # dropout handled by LoRA config
    missing_keys, unexpected_keys = model.load_state_dict(processed_state, strict=False)

    logger.info(f"Loaded {len(processed_state)} keys from checkpoint")
    logger.info(f"Missing keys: {len(missing_keys)}" + str(missing_keys))
    logger.info(f"Unexpected keys: {len(unexpected_keys)}" + str(unexpected_keys))

    # # randomly initialize the head weights with xavier_uniform
    # print(
    #     f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    # )
    model.head.weight.data = nn.init.xavier_uniform_(model.head.weight.data)
    model.head.bias.data = nn.init.zeros_(model.head.bias.data)

    if lora:
        # PEFT expects a HF-like config with .get and attributes it queries. #TODO: check if this is correct
        class _Cfg:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

            # make it dict-like enough for PEFT fallbacks
            def get(self, key, default=None):
                return getattr(self, key, default)

            def to_dict(self):
                return self.__dict__.copy()

        model.config = _Cfg(
            use_return_dict=False,  # your model returns tensors, not dicts
            tie_word_embeddings=False,  # not relevant for LaBraM, but PEFT checks it
        )
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # tells PEFT where/how to inject LoRA adapters and which wrapping logic to use
            # inference_mode=False,
            r=peft_config["r"],
            lora_alpha=peft_config["lora_alpha"],
            lora_dropout=peft_config["lora_dropout"],
            target_modules=peft_config["target_modules"],
            modules_to_save=["head"],
        )
        model = get_peft_model(model, lora_config)

        # ensure head is trainable
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True

    print(
        f"N trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # print("trainable parameters: ", [name for name, param in model.named_parameters() if param.requires_grad])

    return model


def load_labram_with_adapter(adapter_dir: str, device="cpu"):
    # 1) Build your base LaBraM *without* LoRA first (same code you used pre-PEFT)
    base_model = load_labram(lora=False)  # your function above, but with lora=False
    base_model.to(device)

    # 2) (If you added a minimal .config for PEFT before, do it again here)
    class _Cfg:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def get(self, k, default=None):
            return getattr(self, k, default)

        def to_dict(self):
            return self.__dict__.copy()

    # PEFT occasionally checks these flags; harmless for non-HF models:
    base_model.config = _Cfg(
        use_return_dict=False,
        tie_word_embeddings=False,
    )

    # 3) Load the adapter (the folder that has adapter_config.json + adapter_model.safetensors)
    adapter_dir = Path(adapter_dir)
    assert (adapter_dir / "adapter_config.json").exists()
    assert (
        adapter_dir / "adapter_model.safetensors"
    ).exists()  # note: file is "adapter_model", not "adapter_mode"

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir.as_posix(),
        is_trainable=True,  # set True if you want to keep fine-tuning
        adapter_name="default",  # or whatever you used when saving
    )

    return model


def set_partial_finetune_labram(model, verbose=True):
    k = 8
    # 0) freeze all
    for _, p in model.named_parameters():
        p.requires_grad = False

    # find transformer blocks (by name)
    block_map = defaultdict(list)  # idx -> [param names]
    for n, p in model.named_parameters():
        m = re.search(r"(^|\.)(blocks)\.(\d+)\.", n)
        if m:
            idx = int(m.group(3))
            block_map[idx].append(n)

    all_block_ids = sorted(block_map.keys())
    if verbose:
        print(f"[partial-ft] Found {len(all_block_ids)} transformer blocks: {all_block_ids}")
    
    # decide which blocks to unfreeze
    to_unfreeze = set()
    to_unfreeze = set(all_block_ids[-k:])

    name_to_param = dict(model.named_parameters())
    # unfreeze chosen blocks
    for idx in to_unfreeze:
        for pname in block_map.get(idx, []):
            name_to_param[pname].requires_grad = True

    # unfreeze classification head
    head_hits = 0
    for n, p in model.named_parameters():
        ln = n.lower()
        if ("head." in ln):
            p.requires_grad = True
            head_hits += 1
    if verbose:
        print(f"[partial-ft] Unfroze head params: {head_hits} tensors")
        print(f"[partial-ft] Unfroze block ids: {sorted(list(to_unfreeze))}")