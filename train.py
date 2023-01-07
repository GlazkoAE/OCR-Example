import sys
from itertools import groupby

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from colorama import Fore
from tqdm import tqdm

from dataset import CapchaDataset
from model import CRNN

gpu = torch.device("cuda")
train_ds_samples = 1024 * 1000
epochs = 64
train_batch_size = 1024

gru_hidden_size = 128
gru_num_layers = 2
cnn_output_height = 4
cnn_output_width = 32
max_digits_per_sequence = 5

model_save_path = "./checkpoints"


def train_one_epoch(model, criterion, optimizer, data_loader) -> None:
    model.train()
    train_correct = 0
    train_total = 0
    for x_train, y_train in tqdm(
        data_loader,
        position=0,
        leave=True,
        file=sys.stdout,
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
    ):
        batch_size = x_train.shape[0]  # x_train.shape == torch.Size([64, 28, 140])
        x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        optimizer.zero_grad()
        y_pred = model(x_train.cuda())
        y_pred = y_pred.permute(1, 0, 2)  # y_pred.shape == torch.Size([64, 32, 11])
        input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = criterion(y_pred, y_train, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        _, max_index = torch.max(y_pred, dim=2)
        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != train_ds.blank_label]
            )
            if is_correct(prediction, y_train[i], train_ds.blank_label):
                train_correct += 1
            train_total += 1
    print(
        "TRAINING. Correct: ",
        train_correct,
        "/",
        train_total,
        "=",
        train_correct / train_total,
    )


def is_correct(prediction, y_true, blank):
    prediction = prediction.to(torch.int32)
    prediction = prediction[prediction != blank]
    y_true = y_true.to(torch.int32)
    y_true = y_true[y_true != blank]
    return len(prediction) == len(y_true) and torch.all(prediction.eq(y_true))


def evaluate(model, val_loader) -> float:
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for x_val, y_val in tqdm(
            val_loader,
            position=0,
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
        ):
            batch_size = x_val.shape[0]
            x_val = x_val.view(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
            y_pred = model(x_val.cuda()).to("cpu")
            test_hypos = model.beam_search_decoder(y_pred)
            for i in range(batch_size):
                prediction = test_hypos[i][0].tokens.int()
                if is_correct(prediction, y_val[i], train_ds.blank_label):
                    val_correct += 1
                val_total += 1
        acc = val_correct / val_total
        print("TESTING. Correct: ", val_correct, "/", val_total, "=", acc)
    return acc


if __name__ == "__main__":
    train_ds = CapchaDataset((1, max_digits_per_sequence), samples=train_ds_samples)
    test_ds = CapchaDataset((1, max_digits_per_sequence), samples=10000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size)
    val_loader = torch.utils.data.DataLoader(test_ds, batch_size=1)

    model = CRNN(
        cnn_output_height,
        gru_hidden_size,
        gru_num_layers,
        train_ds.num_classes,
        max_digits_per_sequence,
        tokens=train_ds.classes,
    ).to(gpu)
    # model.load_state_dict(torch.load("./checkpoints/epoch_64-acc_0.8919.pt"))

    criterion = nn.CTCLoss(
        blank=train_ds.blank_label, reduction="mean", zero_infinity=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    current_acc = 0
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}")
        train_one_epoch(model, criterion, optimizer, train_loader)
        acc = evaluate(model, val_loader)
        scheduler.step()
        if acc > current_acc:
            current_acc = acc
            model_out_name = model_save_path + f"/epoch_{epoch}-acc_{acc}.pt"
            torch.save(model.to("cpu").state_dict(), model_out_name)
