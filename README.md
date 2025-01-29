# are-16-heads-really-better-than-1

This repository contains the code and data for the project "Are 16 heads really better than 1?".

## Commands

1. **Create the directory for the data:**

   \```
   mkdir -p glue_data/MNLI
   \```

2. **Download the TSV file:**

   \```
   python download_tsv.py
   \```

3. **Run the script for the heads ablation experiment:**

   \```
   ./pytorch-pretrained-BERT/experiments/heads_ablation.sh MNLI
   \```
