# Firestore CSV Command

Dump all `gcp3_cache` documents to CSV.

## Run

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && \
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/mamba.sh && \
mamba activate fin-ai1 && \
python /Users/adamaslan/code/gcp3/.claude/skills/firestore-csv/export_firestore.py
```

Saves CSV to `nu-logs2/gcp3_firestore_TIMESTAMP.csv`.
