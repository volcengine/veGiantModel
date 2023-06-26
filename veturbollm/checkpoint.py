import torch
import os


def save_checkpoint(model, optimizer, lr_scheduler, completed_steps, args):
    save_dir = os.path.join(args.checkpointing.save_dir, f"step-{completed_steps}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "checkpoint.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "completed_steps": completed_steps,
        },
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")
    return save_path
