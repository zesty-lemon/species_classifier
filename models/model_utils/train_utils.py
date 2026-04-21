import torch
import torch.nn as nn
import time


def train_model(model, train_loader, val_loader, device, device_name, epochs=5, lr=0.01, name="Model",
                patience=5, min_delta=1e-4):
    """
    Generic training loop with validation. Returns all metrics for comparison.
    """
    print("------ BEGIN Training Model ------")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # decay the learning rate with cosine annealing
    # Drops the learning rate to its minimum by epoch 30, regardless of the max epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scaler = torch.amp.GradScaler(enabled=(device_name == 'cuda'))  # only enable if running on CUDA
    # ^ the scaler stores numbers in 16 bit instead of 32 bit and dramatically scales up the training time

    print(f"\nTraining {name} for {epochs} epochs...")
    start_time = time.time()

    # Metrics to track
    history = {'train_loss': [],
               'train_acc': [],
               'train_top5_acc': [],
               'val_loss': [],
               'val_acc': [],
               'val_top5_acc': [],
               'learning_rate': []}

    # Used for Early Stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        top5_correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_name):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, top5_pred = outputs.topk(5, dim=1)
            top5_correct += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_top5_acc = 100 * top5_correct / total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type=device_name):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                _, top5_pred = outputs.topk(5, dim=1)
                top5_correct += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct / total
        epoch_val_top5_acc = 100 * top5_correct / total

        # step the scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        current_iteration_status = f"  Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Top-5: {train_top5_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | Val Top-5: {epoch_val_top5_acc:.2f}% | Learning Rate: {current_lr:.6f}%"

        print(current_iteration_status)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_top5_acc'].append(train_top5_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_top5_acc'].append(epoch_val_top5_acc)
        history['learning_rate'].append(current_lr)

        # Early Stopping
        # only count as improvement if val loss drops by more than min_delta
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping triggered at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    duration = time.time() - start_time
    print(
        f"{name} — Final Val Acc: {history['val_acc'][-1]:.2f}%, Val Top-5 Acc: {history['val_top5_acc'][-1]:.2f}%, Time: {duration:.2f}s")
    print("------ END Training Model ------")
    return history, duration