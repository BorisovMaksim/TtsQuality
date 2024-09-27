from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import torchaudio
import torch.nn as nn
from torchaudio.datasets import LIBRISPEECH
import numpy as np


class SimComputer(nn.Module):
    def __init__(
        self, device="cuda:0", model_id="microsoft/wavlm-base-plus-sv", target_sr=16000
    ):
        super().__init__()

        self.model = WavLMForXVector.from_pretrained(model_id)
        self.model.to(device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.target_sr = target_sr
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        self.device = device

    def forward(self, wav_1, sr_1, wav_2, sr_2):
        assert sr_1 == self.target_sr
        assert sr_1 == sr_2

        inputs = self.feature_extractor(
            [wav_1, wav_2], padding=True, return_tensors="pt", sampling_rate=sr_1
        )

        embeddings = self.model(
            input_values=inputs["input_values"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
        ).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

        similarity = self.cosine_sim(embeddings[0], embeddings[1])
        similarity = float(similarity)

        return similarity


if __name__ == "__main__":
    test = LIBRISPEECH(root="./data/", url="test-clean")

    wav_1, sr_1, text_1, _, _, _ = test[1]
    wav_2, sr_2, text_2, _, _, _ = test[2]

    wav_1 = wav_1.numpy()[0]
    wav_2 = wav_2.numpy()[0]

    sim_computer = SimComputer()
    sim = sim_computer(wav_1, sr_1, wav_2, sr_2)

    assert np.allclose(float(sim), 0.9642090797424316)
    print("TEST PASSED")
