import os
import torch
import torchaudio

from transformers import HubertForCTC, Wav2Vec2Processor

from ctc_segmentation import (
    CtcSegmentationParameters,
    ctc_segmentation,
    prepare_text,
    determine_utterance_segments,
)

MODEL_NAME = "facebook/hubert-large-ls960-ft"

print("Loading HuBERT...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = HubertForCTC.from_pretrained(MODEL_NAME)

os.makedirs("PREDICTED_TIMESTAMPS", exist_ok=True)

with open("valid_files.txt", "r", encoding="utf-8") as f:
    file_ids = [line.strip() for line in f if line.strip()]

print(f"Found {len(file_ids)} files")

success = 0
failed = 0

for idx, file_id in enumerate(file_ids, start=1):

    try:

        audio_file = f"WAV_FILES/{file_id}.wav"
        transcript_file = f"TRANSCRIPTS/{file_id}.txt"
        output_file = (
            f"PREDICTED_TIMESTAMPS/"
            f"{file_id}_predicted.txt"
        )

        

        with open(
            transcript_file,
            "r",
            encoding="utf-8"
        ) as f:

            transcript = (
                f.read()
                .strip()
                .upper()
            )

        if transcript == "":
            print(f"[{idx}] Empty transcript: {file_id}")
            continue

    
        waveform, sr = torchaudio.load(audio_file)

        inputs = processor(
            waveform.squeeze(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        log_probs = torch.nn.functional.log_softmax(
            logits,
            dim=-1
        )

        lpz = log_probs[0].cpu().numpy()


        vocab = processor.tokenizer.get_vocab()

        char_list = [""] * len(vocab)

        for token, vocab_id in vocab.items():
            char_list[vocab_id] = token


        config = CtcSegmentationParameters()

        ground_truth = transcript.split()

        if len(ground_truth) == 0:
            continue

        ground_truth_mat, utt_begin_indices = prepare_text(
            config,
            ground_truth,
            char_list
        )

        timings, char_probs, state_list = ctc_segmentation(
            config,
            lpz,
            ground_truth_mat
        )

        segments = determine_utterance_segments(
            config,
            utt_begin_indices,
            char_probs,
            timings,
            ground_truth
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
            f"[{idx}/{len(file_ids)}] "
            f"{file_id} ✓"
        )

    except Exception as e:

        failed += 1

        print(
            f"[{idx}/{len(file_ids)}] "
            f"{file_id} FAILED"
        )

        print(e)

print("\n========================")
print("Processing Complete")
print("========================")
print(f"Success : {success}")
print(f"Failed  : {failed}")