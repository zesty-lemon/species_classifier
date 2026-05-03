import os
from config import constants as c
from matplotlib import pyplot as plt
"""
This file contains code to evaluate model performance
Plotting charts
Generating Reports
"""

def plot_training_curves(history, dataset_name, name="Model"):
    name = name.replace(" ", "_")
    directory = c.PROJECT_ROOT / "graphs_and_stats" / name / dataset_name
    os.makedirs(directory, exist_ok=True)

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
    plt.savefig(f"{directory}/{name.replace(' ', '_')}_training_curves.png", dpi=150)
    plt.show()


def generate_performance_report(model,
                                val_loader,
                                device,
                                device_name,
                                history,
                                dataset_name,
                                name="Model",
                                annotation=""):

    # ---- Generate & Save Report to Directory ----
    name = name.replace(" ", "_")
    directory = c.PROJECT_ROOT / "graphs_and_stats" / name / dataset_name
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(directory, f"{name}_report.txt")

    num_epochs = len(history['train_loss'])

    # ---- Summary stats ----
    best_val_acc = max(history['val_acc'])
    best_val_acc_epoch = history['val_acc'].index(best_val_acc) + 1
    best_val_loss = min(history['val_loss'])
    best_val_loss_epoch = history['val_loss'].index(best_val_loss) + 1
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    overfit_gap = final_train_acc - final_val_acc

    # ---- Write report ----
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{name} Report\n")
        f.write(f"{annotation}\n")
        f.write("======================================\n\n")

        f.write("----- Summary -----\n")
        f.write(f"Best Val Accuracy:  {best_val_acc:.2f}% (epoch {best_val_acc_epoch})\n")
        f.write(f"Best Val Loss:      {best_val_loss:.4f} (epoch {best_val_loss_epoch})\n")
        f.write(f"Final Train Acc:    {final_train_acc:.2f}%\n")
        f.write(f"Final Val Acc:      {final_val_acc:.2f}%\n")
        f.write(f"Overfit Gap:        {overfit_gap:.2f}%\n\n")

        f.write("----- Training History -----\n")
        for i in range(num_epochs):
            train_loss = history['train_loss'][i]
            train_acc = history['train_acc'][i]
            train_top5_acc = history['train_top5_acc'][i]
            epoch_val_loss = history['val_loss'][i]
            epoch_val_acc = history['val_acc'][i]
            epoch_val_top5_acc = history['val_top5_acc'][i]
            lr = history['learning_rate'][i]
            current_iteration_status = f"  Epoch [{i + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Top-5: {train_top5_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | Val Top-5: {epoch_val_top5_acc:.2f}% | LR: {lr}"
            f.write(f"{current_iteration_status}\n")

    print(f"Saved performance report to: {report_path}")