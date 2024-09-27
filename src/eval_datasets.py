from torchaudio.datasets import LIBRISPEECH
import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm
import string


class LibriSpeechTest(Dataset):
    def __init__(
        self,
        root="./data/",
        subset="test-clean",
        download=True,
        target_sr=16000,
        min_len=4,
        max_len=10,
    ) -> None:
        super().__init__()
        self.root = root
        self.subset = subset
        self.download = download
        self.dataset = []
        self.target_sr = target_sr
        self.min_len = min_len
        self.max_len = max_len

    def load_dataset(self):
        self.dataset = LIBRISPEECH(
            root=self.root,
            download=self.download,
            url=self.subset,
        )
        self.indexes = []
        for i, sample in enumerate(self.dataset):
            audio, sr, _, _, _, _ = sample
            dur = audio.shape[-1] / sr
            if dur > self.min_len and dur < self.max_len:
                self.indexes.append(i)

    def __getitem__(self, i: int):
        audio, sr, text, _, _, _ = self.dataset[self.indexes[i]]
        audio = audio.numpy()[0]

        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)

        text = text.lower()
        return audio, self.target_sr, text

    def __len__(self) -> int:
        return len(self.indexes)


class Jenny(Dataset):
    def __init__(
        self, num_samples=1000, target_sr=16000, min_len=4, max_len=10
    ) -> None:
        super().__init__()
        self.dataset = []
        self.num_samples = num_samples
        self.target_sr = target_sr

        self.min_len = min_len
        self.max_len = max_len

    def load_dataset(self):
        dataset = load_dataset("reach-vb/jenny_tts_dataset")["train"]

        indexes = []
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]

            dur = audio.shape[-1] / sr
            if dur > self.min_len and dur < self.max_len:
                indexes.append(i)

        np.random.seed(42)
        random_subset = np.random.choice(indexes, size=self.num_samples, replace=False)
        self.dataset = dataset.select(random_subset)

    def __getitem__(self, i: int):
        sample = self.dataset[i]
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = sample["transcription"]

        if sr != self.target_sr:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), sr, self.target_sr
            )
            audio = audio.numpy()

        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))

        return audio, self.target_sr, text

    def __len__(self) -> int:
        return len(self.dataset)


DATASETS = {
    "librispeech_test_clean": LibriSpeechTest(subset="test-clean"),
    "librispeech_test_other": LibriSpeechTest(subset="test-other"),
    "jenny": Jenny(),
}


if __name__ == "__main__":
    for name, ds in DATASETS.items():
        ds.load_dataset()
        audio, sr, text = ds[0]
        print(
            f"Sample from dataset {name}: {audio.shape}, {type(audio)}, {sr=} with total length of {len(ds)}"
        )
