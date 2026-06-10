from pathlib import Path

TRAIN_DIR = Path("TRAIN")

for dr in ["DR1", "DR2"]:

    dr_path = TRAIN_DIR / dr

    wav_count = len(
        list(dr_path.rglob("*.WAV"))
    )

    txt_count = len(
        list(dr_path.rglob("*.TXT"))
    )

    wrd_count = len(
        list(dr_path.rglob("*.WRD"))
    )

    print(f"\n{dr}")
    print("-" * 20)
    print(f"WAV Files : {wav_count}")
    print(f"TXT Files : {txt_count}")
    print(f"WRD Files : {wrd_count}")

total_wav = len(
    list(TRAIN_DIR.rglob("*.WAV"))
)

print("\n====================")
print(f"TOTAL WAV FILES : {total_wav}")
print("====================")