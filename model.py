import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder


class CRNN(nn.Module):
    def __init__(
        self,
        cnn_output_height,
        gru_hidden_size,
        gru_num_layers,
        num_classes,
        num_digits,
        tokens,
    ):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.norm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)
        self.norm2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.norm3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2)
        self.norm4 = nn.InstanceNorm2d(64)
        self.gru_input_size = cnn_output_height * 64
        self.gru = nn.GRU(
            self.gru_input_size,
            gru_hidden_size,
            gru_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)
        self.num_digits = num_digits
        self.cnn_output_height = cnn_output_height

        self.beam_search_decoder = ctc_decoder(
            lexicon=None,
            tokens=tokens,
            beam_size=100,
            lm_weight=0,
            blank_token=" ",
            sil_token=" ",
            unk_word=" ",
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.leaky_relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        out = self.norm4(out)
        out = F.leaky_relu(out)
        out = out.permute(0, 3, 2, 1)
        out = out.reshape(batch_size, -1, self.gru_input_size)
        out, _ = self.gru(out)
        out = torch.stack(
            [F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])]
        )
        return out
