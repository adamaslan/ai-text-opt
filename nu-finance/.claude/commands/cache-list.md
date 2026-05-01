# List Firestore Cache Keys

List all documents in the gcp3_cache collection with their TTL expiry.

## Run

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/mamba.sh
mamba activate fin-ai1

python3 -c "
from google.cloud import firestore
import os
db = firestore.Client(project=os.environ.get('GCP_PROJECT_ID', 'ttb-lang1'))
docs = list(db.collection('gcp3_cache').stream())
print(f'Total documents: {len(docs)}')
print('')
for doc in docs:
    data = doc.to_dict()
    print(f'  {doc.id}  (expires: {data.get(\"expires_at\", \"no TTL\")})')
"
```
