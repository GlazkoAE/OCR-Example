import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from tqdm import tqdm

from dataset import CapchaDataset
from model import CRNN

gpu = torch.device("cuda")

gru_hidden_size = 128
gru_num_layers = 2
cnn_output_height = 4
cnn_output_width = 32
digits_per_sequence = 5
model_path = "./checkpoints/epoch_64-acc_0.8919.pt"


def test_model(model, test_ds, number_of_test_imgs: int = 10):
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds, batch_size=number_of_test_imgs
    )
    (x_test, y_test) = next(iter(test_loader))
    y_pred = model(
        x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]).cuda()
    ).to("cpu")
    test_hypos = model.beam_search_decoder(y_pred)

    idxs_to_tokens = model.beam_search_decoder.idxs_to_tokens
    for j in tqdm(range(len(x_test))):
        test_tokens = torch.LongTensor([int(item) for item in y_test[j]])
        pred_tokens = test_hypos[j][0].tokens
        y_test_txt = " ".join(idxs_to_tokens(test_tokens))
        y_pred_txt = " ".join(idxs_to_tokens(pred_tokens))

        mpl.rcParams["font.size"] = 8
        plt.imshow(x_test[j], cmap="gray")
        fig = plt.gcf()
        mpl.rcParams["font.size"] = 18
        text_actual = fig.text(x=0.1, y=0.1, s="Actual: " + y_test_txt)
        text_preds = fig.text(x=0.1, y=0.2, s="Predicted: " + y_pred_txt)
        plt.savefig(f"./output/plot_{j}.png")
        text_actual.set_visible(False)
        text_preds.set_visible(False)
        # plt.show()


if __name__ == "__main__":
    test_ds = CapchaDataset((1, digits_per_sequence), samples=1000)

    model = CRNN(
        cnn_output_height,
        gru_hidden_size,
        gru_num_layers,
        test_ds.num_classes,
        digits_per_sequence,
        tokens=test_ds.classes,
    ).to(gpu)
    model.load_state_dict(torch.load(model_path))

    test_model(model, test_ds)
