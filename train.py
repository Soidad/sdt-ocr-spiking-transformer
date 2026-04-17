from py_compile import main

import os
from datetime import datetime
import numpy as np

import torch
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader

from dataset import WikiTextImageDataset
from tokenizer import SimpleTokenizer
from models.sdt_ocr import SpikeDrivenTransformerOCR


# ==============================
# DEVICE
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ==============================
# OUTPUT FOLDER CREATION
# ==============================
run_name = datetime.now().strftime("sdt_ocr_%Y%m%d_%H%M%S")
output_dir = os.path.join("outputs", run_name)
os.makedirs(output_dir, exist_ok=True)

print("Results will be saved in:", output_dir)


# ==============================
# TOKENIZER
# ==============================
tokenizer = SimpleTokenizer()


# ==============================
# DATASETS FIXES
# ==============================
train_dataset = WikiTextImageDataset(
    tokenizer=tokenizer,
    split="train",
    max_samples=50000,
    img_size=(32, 512),
    max_chars=32,
    train=True,
    sources=[
        ("wikimedia/wikipedia", "20231101.fr", "Français"),
    ]
)

val_dataset = WikiTextImageDataset(
    tokenizer=tokenizer,
    split="test",
    max_samples=5000,
    img_size=(32, 512),
    max_chars=32,
    train=False,
    sources=[
        ("wikimedia/wikipedia", "20231101.fr", "Français"),
    ]
)


def collate_fn(batch):
    images = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)

    return images, labels, label_lengths


train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2
)

print("Train batches per epoch:", len(train_loader))
print("Val batches per epoch:", len(val_loader))

# ==============================
# MODEL
# ==============================
model = SpikeDrivenTransformerOCR(
    vocab_size=tokenizer.vocab_size,
    img_size_h=32,
    img_size_w=512,
    patch_size=2,
    embed_dims=128,
    depths=4,
    num_heads=4,
    T=4,
    in_channels=1,
    pooling_stat="1110", # 2*2*2=8 --> img_size_w /8=512/8 --> T=64
).to(device)

criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2   # régularisation
)

best_val_loss = float("inf")
train_losses = []
val_losses = []
Max_step = 4000


# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(50):

    # ======================
    # TRAIN
    # ======================
    model.train()
    total_loss = 0

    for step, (images, labels, label_lengths) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        logits = model(images)
        
        #print("logits shape : ", logits.shape)

        T = logits.size(1)
        input_lengths = torch.full(
            (logits.size(0),),
            T,
            dtype=torch.long
        ).to(device)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)

        loss = criterion(
            log_probs,
            labels,
            input_lengths,
            label_lengths
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        from spikingjelly.clock_driven import functional
        functional.reset_net(model)

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch} | Step {step} | Train Loss {loss.item():.4f}")
            
        if step == Max_step:
            break
            
    num_step = step + 1

    #avg_train_loss = total_loss / len(train_loader)
    avg_train_loss = total_loss / num_step
    train_losses.append(avg_train_loss)

    print(f"Epoch {epoch} | Average Train Loss {avg_train_loss:.4f}")


    # ======================
    # VALIDATION
    # ======================
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for images, labels, label_lengths in val_loader:

            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(images)

            T = logits.size(1)
            input_lengths = torch.full(
                (logits.size(0),),
                T,
                dtype=torch.long
            ).to(device)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.permute(1, 0, 2)

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch} | Validation Loss {avg_val_loss:.4f}")
    
    if epoch % 10 == 0:
        gap = avg_val_loss - avg_train_loss
        print(f"Train/Val gap: {gap:.6f}")


    # ======================
    # SAVE BEST MODEL
    # ======================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, "best_sdt_ocr.pth")
        )
        print(f"New best model saved (Val Loss: {best_val_loss:.4f})")


# ==============================
# SAVE LOSSES
# ==============================
np.save(os.path.join(output_dir, "train_losses.npy"), np.array(train_losses))
np.save(os.path.join(output_dir, "val_losses.npy"), np.array(val_losses))

print("Training finished.")
print("All files saved in:", output_dir)


if __name__ == "__main__":
    main()


