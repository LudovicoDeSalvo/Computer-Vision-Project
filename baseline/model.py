import torch
import torch.nn as nn

class OCRModel(nn.Module):
    """
    Modello OCR ottimizzato (CNN + BiGRU + CTC) per riconoscimento targhe auto.
    """

    def __init__(self, num_classes):
        super().__init__()

        # CNN potenziata
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 x 16 x 64

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 x 8 x 32

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512 x 4 x 32

            nn.Dropout2d(0.3)
        )

        # GRU bidirezionale migliorata
        self.rnn = nn.GRU(
            input_size=512 * 4,
            hidden_size=256,
            num_layers=3,
            bidirectional=True,
            batch_first=True
        )

        # Fully connected per classificazione CTC
        self.fc = nn.Linear(
            in_features=512,  # 256 * 2 bidirezionale
            out_features=num_classes + 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        x = self.cnn(x)  # (B, 512, 4, 32)
        
        # Riorganizzazione per sequenza
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H)
        x = x.view(b, w, -1)  # (B, 32, 512*4)

        # RNN
        x, _ = self.rnn(x)  # (B, 32, 512)

        # FC
        return self.fc(x)  # (B, 32, num_classes + 1)
