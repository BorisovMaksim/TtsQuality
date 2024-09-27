import torch.nn as nn
import torchaudio
from torchaudio.datasets import LIBRISPEECH
import numpy as np
from transformers import Wav2Vec2Processor, HubertForCTC
import torch


class AsrComputer(nn.Module):
    def __init__(
        self,
        device="cuda:0",
        model_id="facebook/hubert-large-ls960-ft",
        target_sr=16000,
    ):
        super().__init__()

        self.model = HubertForCTC.from_pretrained(model_id)
        self.model.to(device)

        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.target_sr = target_sr
        self.device = device

    def forward(self, wav, sr):
        assert sr == self.target_sr
        input_values = self.processor(
            wav, return_tensors="pt", sampling_rate=sr
        ).input_values.to(self.device)

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])

        transcription = transcription.lower()

        return transcription


if __name__ == "__main__":
    test = LIBRISPEECH(root="./data/", url="test-clean")

    wav, sr, text, _, _, _ = test[1]

    wav = wav.numpy()[0]

    asr_computer = AsrComputer()
    text = asr_computer(wav, sr)
    assert text == "stuff it into you his belly counselled him"
    print("TEST PASSEDs")
