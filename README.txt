HuBERT + CTC Annotation Validation Project

Project Pipeline

1. Collect MP3 files (1710 files)
2. Collect transcript files
3. Collect manual annotation files
4. Validate files
   - Remove empty transcripts
   - Remove '#' transcripts
   - Remove corrupted audio
5. Convert MP3 -> WAV (16kHz mono)
6. Run HuBERT + CTC alignment
7. Generate predicted word timestamps
8. Compare with manual annotations
9. Calculate absolute error
10. Calculate mean error
11. Apply threshold 
12. Flag incorrect annotations
13. Generate validation report

Folder Description

MP3_FILES/          Original audio files
WAV_FILES/          Converted WAV files
TRANSCRIPTS/        Transcript text files
ANNOTATIONS/        Manual annotation files


 Scripts

validate_dataset.py
generate_model_timestamps.py
batch_compare_annotation.py
flagged_files.text
excluded_files.text

#for TIMIT
Folder Description:
TRAIN

files Description:

count_timit_files.py
generate_timit_timestamps.py
compare_timit_annotations.py
