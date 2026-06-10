import os
from pathlib import Path

import torch
import torchaudio

from transformers import (
    HubertForCTC,
    Wav2Vec2Processor
)

from ctc_segmentation import (
    CtcSegmentationParameters,
    ctc_segmentation,
    prepare_text,
    determine_utterance_segments,
)

MODEL_NAME = "facebook/hubert-large-ls960-ft"

TRAIN_DIR = Path("TRAIN")
OUTPUT_DIR = Path("TIMIT_PREDICTED_TIMESTAMPS")

OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading HuBERT...")

processor = Wav2Vec2Processor.from_pretrained(
    MODEL_NAME
)

model = HubertForCTC.from_pretrained(
    MODEL_NAME
)

wav_files = list(
    TRAIN_DIR.rglob("*.WAV")
)

print(
    f"Found {len(wav_files)} WAV files"
)

success = 0
failed = 0

for idx, wav_file in enumerate(
    wav_files,
    start=1
):

    try:

        txt_file = wav_file.with_suffix(
            ".TXT"
        )

        if not txt_file.exists():

            print(
                f"Missing TXT: {wav_file}"
            )

            continue

        speaker = wav_file.parent.name
        utt_id = wav_file.stem

        output_file = (
            OUTPUT_DIR /
            f"{speaker}_{utt_id}_predicted.txt"
        )

        with open(
            txt_file,
            "r",
            encoding="utf-8"
        ) as f:

            line = f.read().strip()

        parts = line.split()

        if len(parts) < 3:

            print(
                f"Bad transcript: {txt_file}"
            )

            continue

        transcript = (
            " ".join(parts[2:])
            .replace(".", "")
            .upper()
        )

        if transcript == "":

            continue

        waveform, sr = torchaudio.load(
            wav_file
        )

        if waveform.shape[0] > 1:

            waveform = waveform.mean(
                dim=0,
                keepdim=True
            )

        if sr != 16000:

            waveform = (
                torchaudio.functional.resample(
                    waveform,
                    sr,
                    16000
                )
            )

        inputs = processor(
            waveform.squeeze(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():

            logits = model(
                **inputs
            ).logits

        log_probs = (
            torch.nn.functional.log_softmax(
                logits,
                dim=-1
            )
        )

        lpz = (
            log_probs[0]
            .cpu()
            .numpy()
        )

        vocab = (
            processor
            .tokenizer
            .get_vocab()
        )

        char_list = [
            ""
        ] * len(vocab)

        for token, vocab_id in vocab.items():

            char_list[vocab_id] = token

        config = (
            CtcSegmentationParameters()
        )

        ground_truth = (
            transcript.split()
        )

        if len(
            ground_truth
        ) == 0:

            continue

        (
            ground_truth_mat,
            utt_begin_indices
        ) = prepare_text(
            config,
            ground_truth,
            char_list
        )

        (
            timings,
            char_probs,
            state_list
        ) = ctc_segmentation(
            config,
            lpz,
            ground_truth_mat
        )

        segments = (
            determine_utterance_segments(
                config,
                utt_begin_indices,
                char_probs,
                timings,
                ground_truth
            )
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as out:

            for word, segment in zip(
                ground_truth,
                segments
            ):

                start_time = segment[0]
                end_time = segment[1]

                out.write(
                    f"{start_time:.6f}\t"
                    f"{end_time:.6f}\t"
                    f"{word}\n"
                )

        success += 1

        print(
            f"[{idx}/{len(wav_files)}] "
            f"{speaker}_{utt_id} ✓"
        )

    except Exception as e:

        failed += 1

        print(
            f"[{idx}/{len(wav_files)}] "
            f"{wav_file.name} FAILED"
        )

        print(e)

print("\n====================")
print("Processing Complete")
print("====================")
print(f"Success : {success}")
print(f"Failed  : {failed}")