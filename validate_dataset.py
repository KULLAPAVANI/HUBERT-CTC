import os

WAV_FOLDER = "WAV_FILES"
TRANSCRIPT_FOLDER = "TRANSCRIPTS"
ANNOTATION_FOLDER = "ANNOTATIONS"

valid_files = []
invalid_files = []

wav_files = [f for f in os.listdir(WAV_FOLDER)
             if f.endswith(".wav")]

for wav_file in wav_files:

    base_name = os.path.splitext(wav_file)[0]

    transcript_file = os.path.join(
        TRANSCRIPT_FOLDER,
        base_name + ".txt"
    )

    annotation_file = os.path.join(
        ANNOTATION_FOLDER,
        base_name + "_Annotated.txt"
    )

    if not os.path.exists(transcript_file):
        invalid_files.append(base_name)
        continue

    if not os.path.exists(annotation_file):
        invalid_files.append(base_name)
        continue

    with open(transcript_file,
              "r",
              encoding="utf-8") as f:

        text = f.read().strip()

    if text == "" or text == "#":
        invalid_files.append(base_name)
        continue

    valid_files.append(base_name)

print("Valid Files:", len(valid_files))
print("Invalid Files:", len(invalid_files))

with open("valid_files.txt", "w") as f:
    for item in valid_files:
        f.write(item + "\n")