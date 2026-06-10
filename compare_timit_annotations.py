import os
import csv

TRAIN_DIR = "TRAIN"
PREDICTED_DIR = "TIMIT_PREDICTED_TIMESTAMPS"

files_compared = 0
red_flag_files = 0

total_words = 0
flagged_words = 0

total_start_error = 0.0
total_end_error = 0.0

all_file_data = []

# -----------------------------
# Read WRD File
# -----------------------------

def read_wrd_file(path):

    words = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 3:
                continue

            start = int(parts[0]) / 16000
            end = int(parts[1]) / 16000

            word = parts[2].upper()

            words.append(
                (word, start, end)
            )

    return words


# -----------------------------
# Read Predicted File
# -----------------------------

def read_predicted_file(path):

    words = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 3:
                continue

            word = parts[2].upper()

            if word.isdigit():
                continue

            words.append(
                (
                    word,
                    float(parts[0]),
                    float(parts[1])
                )
            )

    return words


# -----------------------------
# PASS 1
# -----------------------------

for root, dirs, files in os.walk(TRAIN_DIR):

    for file in files:

        if not file.endswith(".WRD"):
            continue

        wrd_file = os.path.join(
            root,
            file
        )

        speaker = os.path.basename(root)

        utt_id = os.path.splitext(file)[0]

        predicted_file = os.path.join(
            PREDICTED_DIR,
            f"{speaker}_{utt_id}_predicted.txt"
        )

        if not os.path.exists(predicted_file):
            continue

        human = read_wrd_file(
            wrd_file
        )

        predicted = read_predicted_file(
            predicted_file
        )

        n = min(
            len(human),
            len(predicted)
        )

        if n == 0:
            continue

        files_compared += 1

        file_errors = []

        for i in range(n):

            _, h_start, h_end = human[i]
            _, p_start, p_end = predicted[i]

            start_error = abs(
                h_start - p_start
            )

            end_error = abs(
                h_end - p_end
            )

            combined_error_ms = (
                (
                    start_error +
                    end_error
                ) / 2
            ) * 1000

            total_start_error += (
                start_error
            )

            total_end_error += (
                end_error
            )

            file_errors.append(
                combined_error_ms
            )

            total_words += 1

        all_file_data.append(
            (
                speaker,
                utt_id,
                file_errors
            )
        )

# -----------------------------
# Mean Errors
# -----------------------------

mean_start_error = (
    total_start_error /
    total_words
) * 1000

mean_end_error = (
    total_end_error /
    total_words
) * 1000

threshold_ms = (
    max(
        mean_start_error,
        mean_end_error
    ) * 2
)

# -----------------------------
# PASS 2
# -----------------------------

report_rows = []
flagged_files = []

for speaker, utt_id, errors in all_file_data:

    file_flagged = False

    for err in errors:

        if err > threshold_ms:

            flagged_words += 1
            file_flagged = True

    if file_flagged:

        red_flag_files += 1

        flagged_files.append(
            f"{speaker}_{utt_id}"
        )

        status = "RED_FLAG"

    else:

        status = "VALID"

    report_rows.append(
        [
            speaker,
            utt_id,
            status
        ]
    )

percentage_flagged = (
    flagged_words /
    total_words
) * 100

# -----------------------------
# CSV Report
# -----------------------------

with open(
    "timit_validation_report.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "speaker",
            "utterance",
            "status"
        ]
    )

    writer.writerows(
        report_rows
    )

# -----------------------------
# Flagged Files
# -----------------------------

with open(
    "timit_flagged_files.txt",
    "w",
    encoding="utf-8"
) as f:

    for file_id in flagged_files:

        f.write(
            file_id + "\n"
        )

# -----------------------------
# Results
# -----------------------------

print("\n========== RESULTS ==========\n")

print(
    f"Files compared      : "
    f"{files_compared}"
)

print(
    f"Red flag files      : "
    f"{red_flag_files}"
)

print(
    f"Total words         : "
    f"{total_words}"
)

print(
    f"Flagged words       : "
    f"{flagged_words}"
)




print()

print(
    f"Mean Start Error    : "
    f"{mean_start_error:.2f} ms"
)

print(
    f"Mean End Error      : "
    f"{mean_end_error:.2f} ms"
)

print()

print(
    f"Threshold Used      : "
    f"{threshold_ms:.2f} ms"
)

print()

print(
    "Created: "
    "timit_validation_report.csv"
)

print(
    "Created: "
    "timit_flagged_files.txt"
)