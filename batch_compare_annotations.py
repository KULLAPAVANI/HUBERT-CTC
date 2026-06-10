import os
import csv

ANNOTATIONS_DIR = "ANNOTATIONS"
PREDICTED_DIR = "PREDICTED_TIMESTAMPS"

with open("excluded_files.txt", "r") as f:
    excluded = {
        line.strip()
        for line in f
        if line.strip()
    }


def read_file(path):

    words = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 3:
                continue

            words.append(
                (
                    parts[2].upper(),
                    float(parts[0]),
                    float(parts[1])
                )
            )

    return words


files_compared = 0
total_words = 0

total_start_error = 0.0
total_end_error = 0.0

all_word_errors = []
file_word_errors = {}

for file in os.listdir(ANNOTATIONS_DIR):

    if not file.endswith("_Annotated.txt"):
        continue

    file_id = file.replace(
        "_Annotated.txt",
        ""
    )

    if file_id in excluded:
        continue

    manual_file = os.path.join(
        ANNOTATIONS_DIR,
        file
    )

    predicted_file = os.path.join(
        PREDICTED_DIR,
        f"{file_id}_predicted.txt"
    )

    if not os.path.exists(predicted_file):
        continue

    manual = read_file(manual_file)
    predicted = read_file(predicted_file)

    n = min(
        len(manual),
        len(predicted)
    )

    if n == 0:
        continue

    files_compared += 1

    file_word_errors[file_id] = []

    for i in range(n):

        _, m_start, m_end = manual[i]
        _, p_start, p_end = predicted[i]

        start_error = abs(
            m_start - p_start
        )

        end_error = abs(
            m_end - p_end
        )

        combined_error_ms = (
            (
                start_error +
                end_error
            ) / 2
        ) * 1000

        total_start_error += start_error
        total_end_error += end_error

        all_word_errors.append(
            combined_error_ms
        )

        file_word_errors[file_id].append(
            combined_error_ms
        )

        total_words += 1


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


flagged_words = 0
red_flag_files = 0

flagged_files = []
report_rows = []

for file_id, errors in file_word_errors.items():

    file_flagged = False

    for err in errors:

        if err > threshold_ms:

            flagged_words += 1
            file_flagged = True

    if file_flagged:

        red_flag_files += 1

        flagged_files.append(
            file_id
        )

        status = "RED_FLAG"

    else:

        status = "VALID"

    report_rows.append([
        file_id,
        len(errors),
        round(max(errors), 2),
        status
    ])

percentage_flagged = (
    flagged_words /
    total_words
) * 100


with open(
    "validation_report.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "file_id",
        "num_words",
        "max_error_ms",
        "status"
    ])

    writer.writerows(report_rows)

# ----------------------------------
# flagged_files.txt
# ----------------------------------

with open(
    "flagged_files.txt",
    "w",
    encoding="utf-8"
) as f:

    for file_id in flagged_files:

        f.write(file_id + "\n")

# ----------------------------------
# RESULTS
# ----------------------------------

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

print("Created:")
print("validation_report.csv")
print("flagged_files.txt")