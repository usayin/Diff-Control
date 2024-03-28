from lightning.pytorch.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from .StableDiffusionControlnetPipeline import StableDiffusionControlnetPipeline
from .Decoder import Decoder

import lightning as L
import torchmetrics


@torch.no_grad()
def acc(decoded_tensor, target_tensor):
    if decoded_tensor.shape != target_tensor.shape:
        raise ValueError("Tensors must have the same shape.")

    num_matching = (decoded_tensor == target_tensor).sum().item()

    total_elements = decoded_tensor.numel()

    matching_rate = 1.0 * num_matching / total_elements

    return matching_rate


class StegoPipeline(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_pipeline = StableDiffusionControlnetPipeline()
        self.diffusion_pipeline.turn_on_train_mode()
        self.decoder = Decoder()

        # enable partial loading 
        self.strict_loading = False

    
    def state_dict(self):
        state_dict = {}
        for k, v in super().state_dict().items():
            if k.startswith("decoder.") or k.startswith("diffusion_pipeline.controlnet."):
                state_dict[k] = v
        return state_dict


    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        # training_step defines the train loop.
        message, target, prompts = batch

        # message don't track gradient cause it is loaded from pt file
        message.requires_grad_(True)
        target.requires_grad_(True)

        stego_images = self.diffusion_pipeline(
            secret_inputs=message,
            prompts=list(prompts)
        )
        decoded_output = self.decoder(stego_images)
        loss = nn.MSELoss(reduction='mean')(decoded_output, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        message, target, prompts = batch
        stego_images = self.diffusion_pipeline(
            secret_inputs=message,
            prompts=list(prompts)
        )
        decoded_output = self.decoder(stego_images)
        val_loss = F.mse_loss(decoded_output, target)
        val_acc = acc(decoded_output, target)
        metrics = {"val_loss": val_loss, "val_acc": val_acc}
        self.log_dict(metrics)

    
    def test_step(self, batch, batch_idx):
        message, target, prompts = batch
        stego_images = self.diffusion_pipeline(
            secret_inputs=message,
            prompts=list(prompts)
        )
        decoded_output = self.decoder(stego_images)
        test_acc = torchmetrics.Accuracy(decoded_output, target)
        self.log("test_acc", test_acc)


    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"params": self.decoder.parameters()}, {"params": self.diffusion_pipeline.controlnet.parameters()}],
            lr=1e-4,
        )

        return optimizer


if __name__ == "__main__":
    secret_input = torch.randn((3, 3, 64, 64)).to("cuda")
    prompts = ["a cat"] * 3
    negative_prompts = [""] * 3
    seed = 1

    model = StegoPipeline()    
    model.train()

    model.decoder.to("cuda")

    decoded_output = model(
        secret_input,
        prompts=prompts,
        negative_prompts=negative_prompts,
        seed=1
    )

    print(secret_input.shape)
    print(decoded_output.shape)
