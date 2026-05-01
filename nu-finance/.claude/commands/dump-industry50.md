# Dump Industry50 Data

Export full industry50 cached data from Firestore `gcp3_cache` to CSV — all dates, all industries, with returns and 52W hi/lo.

## Steps

1. Read all `industry50:*` documents from Firestore `gcp3_cache`
2. Flatten `value.industries` into rows with price, change, 13 return periods, 52W hi/lo
3. Export leaders/laggards per date as a separate CSV
4. Report row counts and file paths

## Run

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && \
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/mamba.sh && \
mamba activate fin-ai1 && \
GCP_PROJECT_ID=$(gcloud config get-value project 2>/dev/null) python -c "
import csv
from datetime import datetime
from google.cloud import firestore

db = firestore.Client()
coll = db.collection('gcp3_cache')

docs = [d for d in coll.stream() if d.id.startswith('industry_data:')]
docs.sort(key=lambda d: d.id)

return_periods = ['1d','3d','1w','2w','3w','1m','3m','6m','ytd','1y','2y','5y','10y']
header = ['cache_date','industry','etf','sector','price','change','change_pct','source'] + [f'return_{p}' for p in return_periods] + ['52w_high','52w_low']

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
outpath = f'/Users/adamaslan/code/nu-logs2/industry50_full_{ts}.csv'
row_count = 0
with open(outpath, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    for doc in docs:
        data = doc.to_dict()
        value = data.get('value', data)
        cache_date = doc.id.replace('industry_data:', '')
        industries = value.get('industries', {})
        for name, ind in sorted(industries.items()):
            returns = ind.get('returns', {})
            row = [
                cache_date, name, ind.get('etf',''), ind.get('sector',''),
                ind.get('price',''), ind.get('change',''), ind.get('change_pct',''),
                ind.get('source',''),
            ]
            row += [returns.get(p, '') for p in return_periods]
            row += [ind.get('52w_high',''), ind.get('52w_low','')]
            w.writerow(row)
            row_count += 1

outpath2 = f'/Users/adamaslan/code/nu-logs2/industry50_leaders_laggards_{ts}.csv'
ll_count = 0
with open(outpath2, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['cache_date','type','rank','industry','etf','sector','price','change','change_pct'])
    for doc in docs:
        data = doc.to_dict()
        value = data.get('value', data)
        cache_date = doc.id.replace('industry_data:', '')
        for ltype in ['leaders','laggards']:
            for i, item in enumerate(value.get(ltype, []), 1):
                w.writerow([cache_date, ltype, i, item.get('industry',''), item.get('etf',''), item.get('sector',''), item.get('price',''), item.get('change',''), item.get('change_pct','')])
                ll_count += 1

print(f'{len(docs)} dates x industries = {row_count} rows')
print(f'{outpath}')
print(f'{ll_count} leader/laggard entries')
print(f'{outpath2}')
"
```

## What to Report

- Number of cached dates and total rows
- Full file paths for both CSVs
- Flag any dates missing returns data (returns columns all empty)
- Flag if no industry50 documents exist (backend hasn't cached yet)
