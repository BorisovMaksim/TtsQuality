import os

os.environ["CURL_CA_BUNDLE"] = ""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from argparse import ArgumentParser

from jiwer import wer, cer
import json
from pathlib import Path
from tqdm import tqdm
import torchaudio
import numpy as np
import shutil

from fadtk import FrechetAudioDistance
from fadtk.model_loader import W2V2Model

from pymcd.mcd import Calculate_MCD

from src.compute.asr import AsrComputer
from src.compute.sim import SimComputer
from src.eval_datasets import DATASETS


class QualityComputer(nn.Module):
    def __init__(
        self,
        device="cuda:0",
        description="Jenny speaks at an average pace with an animated delivery in a very confined sounding environment with clear audio quality.",
        target_sr=16000,
    ):
        super().__init__()

        self.device = device
        self.target_sr = target_sr
        self.description = description
        self.model = None
        self.tokenizer = None
        self.asr_computer = None
        self.sim_computer = None
        self.fad = None
        self.fad_model = None
        self.mcd_toolbox = None

    def load_model(self):
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-mini-jenny-30H"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "parler-tts/parler-tts-mini-jenny-30H"
        )
        self.input_ids = self.tokenizer(
            self.description, return_tensors="pt"
        ).input_ids.to(self.device)
        self.sr = self.model.config.sampling_rate

    def load_assessment_models(self):
        self.asr_computer = AsrComputer(device=self.device)
        self.sim_computer = SimComputer(device=self.device)
        self.fad_model = W2V2Model(size="base", layer=12)
        self.fad = FrechetAudioDistance(
            self.fad_model, audio_load_worker=4, load_model=True
        )
        self.mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")

    def generate_wav(
        self,
        prompt="Hey, how are you doing today? My name is Jenny, and I'm here to help you with any questions you have.",
    ):
        if self.model is None:
            self.load_model()

        prompt = prompt.upper()
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        generation = self.model.generate(
            input_ids=self.input_ids, prompt_input_ids=prompt_input_ids
        )
        audio = torchaudio.functional.resample(generation, self.sr, self.target_sr)
        audio_arr = audio.squeeze().cpu().numpy()

        if (audio_arr == 0).all():
            logger.info(f"Regenerating wav with text = {prompt}")
            audio_arr, _ = self.generate_wav(prompt)
        return audio_arr, self.target_sr

    def forward(self, wav, sr, text, gen_wav, gen_sr):
        if self.asr_computer is None:
            self.load_assessment_models()

        logger.info(f"Original {text}")
        gen_text = self.asr_computer(gen_wav, gen_sr)
        asr_text = self.asr_computer(wav, sr)

        logger.info(f"Generated {gen_text}\nTranscribed {asr_text}")
        real_wer = wer(text, asr_text)
        real_cer = cer(text, asr_text)

        gen_wer = wer(text, gen_text)
        gen_cer = cer(text, gen_text)

        sim = self.sim_computer(wav, sr, gen_wav, gen_sr)

        wav_len = wav.shape[0]
        sim_gt = self.sim_computer(wav[: wav_len // 2], sr, wav[wav_len // 2 :], sr)

        metrics = {
            "real_wer": round(real_wer, 4),
            "real_cer": round(real_cer, 4),
            "gen_wer": round(gen_wer, 4),
            "gen_cer": round(gen_cer, 4),
            "sim": round(sim, 4),
            "sim_gt": round(sim_gt, 4),
        }
        logger.info(metrics)
        return metrics

    def compute_fad(self, saved_paths, saved_gt_paths):

        wavs_path = saved_paths[0].parent
        gt_wavs_path = saved_gt_paths[0].parent

        logger.info("Caching embedding files...")
        for f in tqdm(saved_paths):
            self.fad.cache_embedding_file(f)

        for f in tqdm(saved_gt_paths):
            self.fad.cache_embedding_file(f)

        score = self.fad.score(wavs_path, gt_wavs_path)

        np.random.seed(42)
        np.random.shuffle(saved_gt_paths)

        wav_half_1_save_path = wavs_path.parent / "gt_wavs_half_1"
        wav_half_2_save_path = wavs_path.parent / "gt_wavs_half_2"

        wav_half_1_save_path.mkdir(exist_ok=True, parents=True)
        wav_half_2_save_path.mkdir(exist_ok=True, parents=True)

        for i, f in enumerate(saved_gt_paths[: len(saved_gt_paths) // 2]):
            gt_save_path = wav_half_1_save_path / f"{i}.wav"
            if not gt_save_path.exists():
                shutil.copyfile(f, gt_save_path)
            self.fad.cache_embedding_file(gt_save_path)

        for i, f in enumerate(saved_gt_paths[len(saved_gt_paths) // 2 :]):
            gt_save_path = wav_half_2_save_path / f"{i}.wav"
            if not gt_save_path.exists():
                shutil.copyfile(f, gt_save_path)
            self.fad.cache_embedding_file(gt_save_path)

        gt_score = self.fad.score(wav_half_1_save_path, wav_half_2_save_path)

        return score, gt_score


def main(device, description, dataset_name, save_dir):
    logging.basicConfig(filename="tts.log", level=logging.INFO)

    dataset = DATASETS[dataset_name]
    dataset.load_dataset()

    computer = QualityComputer(device, description)

    metrics_path = (
        Path(save_dir)
        / dataset_name
        / description.replace(" ", "_").replace(".", "")[:100]
        / "metrics.json"
    )
    wavs_path = metrics_path.parent / "wavs"
    gt_wavs_path = metrics_path.parent / "gt_wavs"

    wavs_path.mkdir(exist_ok=True, parents=True)
    gt_wavs_path.mkdir(exist_ok=True, parents=True)

    saved_paths = []
    saved_gt_paths = []

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        save_path = wavs_path / f"{i}.wav"
        save_path_gt = gt_wavs_path / f"{i}.wav"

        saved_paths.append(save_path)
        saved_gt_paths.append(save_path_gt)

        audio, sr, text = sample

        if not save_path_gt.exists():
            sf.write(save_path_gt, audio, sr)

        if not save_path.exists():
            with torch.no_grad():
                gen_wav, gen_sr = computer.generate_wav(text)
            sf.write(save_path, gen_wav, gen_sr)

    torch.cuda.empty_cache()

    res = {}

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        audio, sr, text = sample
        gen_wav, gen_sr = sf.read(saved_paths[i])
        with torch.no_grad():
            metrics = computer(audio, sr, text, gen_wav, gen_sr)
        if len(res) == 0:
            res = {k: [v] for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                res[k] += [v]

    for k, v in metrics.items():
        res[k] = np.mean(v)

    score, gt_score = computer.compute_fad(saved_paths, saved_gt_paths)

    mcd_scores = []
    for f_gen, f_real in tqdm(zip(saved_paths, saved_gt_paths), total=len(saved_paths)):
        mcd_value = computer.mcd_toolbox.calculate_mcd(f_real, f_gen)
        mcd_scores.append(mcd_value)

    res["fad_score"] = round(score, 4)
    res["fad_score_gt"] = round(gt_score, 4)
    res["mcd_score"] = round(np.mean(mcd_scores), 4)

    with open(metrics_path, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="librispeech_test_clean",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Jenny speaks at an average pace with an animated delivery in a very confined sounding environment with clear audio quality.",
    )
    parser.add_argument("--save_dir", type=str, default="./results")

    args = parser.parse_args()
    main(**vars(args))
