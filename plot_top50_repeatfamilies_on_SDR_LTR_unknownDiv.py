## this script plots the %div top 50 repeat families for ltr unknown for sex chromosomes

import os, re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
from collections import defaultdict
import re
from collections import defaultdict
import pandas as pd
from plotnine import *
import joypy
import matplotlib.pyplot as plt

def parse_repeatmasker_out(file_path):
    unassignedcount = defaultdict(int)
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('SW') or line.startswith('score') or line.startswith('---'):
                continue

            parts = re.split(r'\s+', line)
            if len(parts) < 14:
                continue

            try:
                chrom = parts[4]
                start = int(parts[5])
                end = int(parts[6])
                divergence = float(parts[1])
                repeat_name = parts[9]
                class_family = parts[10]
                midpoint = (start + end) // 2
                region = None
                if chrom == 'Y':
                    if midpoint > 94e6:
                        region = 'Y-PAR'
                    elif midpoint <= 94_000_000:
                        region = 'SDR'
                elif chrom == 'X':
                    if midpoint > 205e6:
                        region = 'X-PAR'
                    elif midpoint <= 205_000_000:
                        region = 'X-NR'
                else:
                    continue

                if ('Caulimovirus' in class_family or
                    'Low_complexity' in class_family or
                    'LINE/L1' in class_family):
                    continue
                if 'DTA' in class_family or 'hAT' in class_family:
                    class_family = 'hAT'
                elif 'DTC' in class_family:
                    class_family = 'CACTA'
                elif 'DTM' in class_family or "DNA/MULE-MuDR" in class_family:
                    class_family = 'MULE-MuDR'
                elif 'DTT' in class_family:
                    class_family = 'Tc1-Mariner'
                elif 'DTH' in class_family or 'Harbinger' in class_family:
                    class_family = 'PIF/Harbinger'
                elif 'Helitron' in class_family:
                    class_family = 'Helitron'
                elif 'Gypsy' in class_family:
                    class_family = 'Ty3/Gypsy'
                elif 'Copia' in class_family:
                    class_family = 'Ty1/Copia'
                elif 'LTR/unknown' in class_family:
                    class_family = 'LTR/unknown'
                else:
                    continue
                #if class_family not in ['Ty3/Gypsy', 'Ty1/Copia', 'LTR/unknown']:                                                                                                                                                                                                                                                                                                                                                                                                                    
                fam = class_family
                if 'pan' not in repeat_name:
                    records.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'repeat_name': repeat_name,
                        'te_class': fam,
                        'region': region,
                        'divergence': divergence
                })
            except Exception:
                continue

    return pd.DataFrame(records)


def get_counts(df):
    counts = (
        df.groupby(['repeat_name', 'region', 'te_class'])
        .size()
        .reset_index(name='count')
    )
    return counts.pivot_table(index=['repeat_name', 'te_class'], columns='region', values='count', fill_value=0).reset_index()

region_sizes = {
    'SDR': 94_000_000,
    'X-NR': 205_000_000,
    'X-PAR': 230e6-205e6,
    'Y-PAR': 117e6-94e6,
}
def normalize_counts(df):
    for region in ['SDR', 'X-NR', 'X-PAR', 'Y-PAR']:
        if region in df.columns:
            size_mb = region_sizes.get(region, 1_000_000) / 1_000_000
            #df[region] = df[region] / 1e6
            
            df[region] = df[region] / size_mb
    return df


file_path = 'ragtag19058m.fasta.mod.panEDTA.out'  # <-- change to your file path                                                                  
df = parse_repeatmasker_out(file_path)
summary_df = get_counts(df)

for col in ['SDR', 'X-NR', 'X-PAR', 'Y-PAR']:
    if col not in summary_df.columns:
        summary_df[col] = 0

summary_df = normalize_counts(summary_df)

import pandas as pd
import joypy
import matplotlib.pyplot as plt

# Filter only Ty3/Gypsy
ty3_df = summary_df[summary_df['te_class'] == 'Ty3/Gypsy'].copy()

# Ensure SDR column exists
if 'SDR' not in ty3_df.columns:
    ty3_df['SDR'] = 0

# Get top 20 families by SDR counts
#top20_families = ty3_df.sort_values('SDR', ascending=False).head(20)['repeat_name'].tolist()

top20_families = ty3_df.sort_values('SDR', ascending=False)['repeat_name'].tolist()

# Filter original df for those families
top20_df = df[df['repeat_name'].isin(top20_families) & (df['te_class'] == 'Ty3/Gypsy')].copy()

# Make sure divergence column is numeric
top20_df['divergence'] = pd.to_numeric(top20_df['divergence'], errors='coerce')


import pandas as pd
import joypy
import matplotlib.pyplot as plt

# Filter Ty3/Gypsy
ty3_df = df[df['te_class'] == 'LTR/unknown'].copy()

# Ensure divergence is numeric
ty3_df['divergence'] = pd.to_numeric(ty3_df['divergence'], errors='coerce')


#sdr_counts = ty3_df[ty3_df['region'] == 'SDR'].groupby('repeat_name').size()
#sdr_counts = ty3_df[ty3_df.groupby('repeat_name').size()
# Get top 20 families by SDR counts
#top20_families = sdr_counts.sort_values(ascending=False).head(50).index.tolist()
#top20_families = sdr_counts.sort_values(ascending=False).index.tolist()
#top20_families = ty3_df.sort_values('SDR', ascending=False)['repeat_name'].tolist()

#top20_df = ty3_df[ty3_df['repeat_name'].isin(top20_families)].copy()

repeat_counts = ty3_df.groupby('repeat_name').size().reset_index(name='count')

# Sort by count descending
repeat_counts = repeat_counts.sort_values(by='count', ascending=False)

# Select top 50
top50_repeats = repeat_counts.head(50)

# Optional: reset index
top50_repeats = top50_repeats.reset_index(drop=True)

print(top50_repeats)

top50_repeat_names = top50_repeats['repeat_name'].tolist()

# Subset original dataframe for only top 50 repeats
top50_df = ty3_df[ty3_df['repeat_name'].isin(top50_repeat_names)]

# Make ridgeline plot of divergence by region
fig, axes = joypy.joyplot(
    top50_df,
    by='region',
    column='divergence',
    kind='kde',
    fill=True,
    overlap=1,
    linewidth=0.2,
    color="#b2df8a"
)


'''
fig, axes = joypy.joyplot(
    top50_repeats,
    by='region',
    column='divergence',
    kind='kde',
    fill=True,
    overlap=1,
    linewidth=1,
    color="#8856a7"
)
'''

fig.set_size_inches(2, 2)  # width, height in inches
plt.xlim(0, top50_df['divergence'].max())
plt.xlabel("% Divergence")

plt.savefig("top50DivLTR_unknown.pdf", bbox_inches="tight")



