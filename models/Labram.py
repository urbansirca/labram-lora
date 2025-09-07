from torcheeg.models.transformer.labram import LaBraM
import torch
from collections import OrderedDict
import pathlib
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
import yaml


def load_labram(lora=True, peft_config=None):
    # Load checkpoint first
    MODEL_PATH = (
        pathlib.Path(__file__).parent
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
    model = LaBraM.base_patch200_200(
        num_classes=2
    )  # TODO: not sure if the head should be initialized here or added on top of the model
    missing_keys, unexpected_keys = model.load_state_dict(processed_state, strict=False)

    print(f"Loaded {len(processed_state)} keys from checkpoint")
    print(f"Missing keys: {len(missing_keys)}", missing_keys)
    print(f"Unexpected keys: {len(unexpected_keys)}", unexpected_keys)

    # # randomly initialize the head weights with xavier_uniform
    # print(
    #     f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    # )
    model.head.weight.data = nn.init.xavier_uniform_(model.head.weight.data)
    model.head.bias.data = nn.init.zeros_(model.head.bias.data)

    if lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # tells PEFT where/how to inject LoRA adapters and which wrapping logic to use
            inference_mode=False,
            r=peft_config["r"],
            lora_alpha=peft_config["lora_alpha"],
            lora_dropout=peft_config["lora_dropout"],
            target_modules=peft_config["target_modules"],
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


if __name__ == "__main__":
    with open("hyperparameters.yaml", "r") as f:
        hyperparameters = yaml.safe_load(f)

    print(f"Hyperparameters: {hyperparameters}")
    model = load_labram(
        lora=hyperparameters["labram"]["lora"],
        peft_config=hyperparameters["peft_config"],
    )
