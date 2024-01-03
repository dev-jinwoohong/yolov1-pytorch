import torch
import torch.optim as optim
from model import YoloV1, YoloVGGV1
from tqdm import tqdm
from dataset import VOCDataset
from loss import YoloLoss
from utils import get_bboxes, mean_average_precision, save_checkpoint

# Hyperparameters
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = True
EPOCHS = 1000
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOAD_MODEL_FILE = "overfit.pth.tar"


def train_fn(train_loader, model, optimizer, loss_fn):
    mean_loss = []

    for batch_idx, (x, y) in enumerate(tqdm(train_loader, leave=True)):
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)

        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    print("It works on", DEVICE)

    model = YoloVGGV1().to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    train_dataset = VOCDataset(root_dir="", image="JPEGImages",
                               annotation="Annotations", is_debug=False)

    test_dataset = VOCDataset(root_dir="", image="JPEGImages",
                              annotation="Annotations", is_debug=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        sampler=None,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, sampler=None,
    )

    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)


if __name__ == "__main__":
    main()
