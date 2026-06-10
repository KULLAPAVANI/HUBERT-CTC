# x.py

from pathlib import Path
ANNOTATION_DIR = "annotations"

annotation_path = Path(ANNOTATION_DIR)

hash_files = []

for file in annotation_path.iterdir():
    if file.is_file():
        try:
            content = file.read_text(encoding="utf-8")

            if "#" in content:
                hash_files.append(file.name)

        except Exception as e:
            print(f"Error reading {file.name}: {e}")

print("\n===== FILES CONTAINING '#' =====")
for filename in hash_files:
    print(filename)

print(f"\nTotal files containing '#': {len(hash_files)}")