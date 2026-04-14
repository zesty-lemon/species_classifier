from matplotlib import pyplot as plt
"""
Expecting History to be defined as
history = {'train_loss': [],
           'train_acc': [],
           'train_top5_acc': [],
           'val_loss': [],
           'val_acc': [],
           'val_top5_acc': [],
           'learning_rate': []}
"""

def plot_training_curves(history, name="Model"):

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{name} — Loss per Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Top-5 Accuracy ---
    ax2.plot(epochs, history["train_top5_acc"], "o-", label="Train Top-5 Acc")
    ax2.plot(epochs, history["val_top5_acc"], "s-", label="Val Top-5 Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-5 Accuracy (%)")
    ax2.set_title(f"{name} — Top-5 Accuracy per Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_')}_training_curves.png", dpi=150)
    plt.show()