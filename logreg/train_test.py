import pandas as pd
import torch


def train(model, criterion, optimizer, train_loader, num_epochs, input_size):
    # Train the model
    model.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Reshape images to (batch_size, input_size)
            images = images.reshape(-1, input_size)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )


def test(model, test_loader, input_size, pred_path):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        predictions = []
        for images, labels in test_loader:
            images = images.reshape(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            prediction = [x.item() for x in predicted]
            predictions.extend(prediction)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        predictions_frame = pd.DataFrame(data={"labels": predictions})
        predictions_frame.to_csv(pred_path)

        print(
            "Accuracy of the model on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )
