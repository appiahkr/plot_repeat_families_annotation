import pandas as pd
import numpy as np
import re
from plotnine import *
from collections import defaultdict

# === PARSE FUNCTION ===
def parse_repeatmasker_out(file_path):
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
                divergence = float(parts[1])
                chrom = parts[4]
                start = int(parts[5])
                end = int(parts[6])
                repeat_name = parts[9]
                class_family = parts[10]
                midpoint = (start + end) // 2
                length = end - start + 1

                region = None
                if chrom == 'Y':
                    region = 'Y-PAR' if midpoint > 94e6 else 'SDR'
                elif chrom == 'X':
                    region = 'X-PAR' if midpoint > 205e6 else 'X-NR'
                else:
                    continue

                if 'Caulimovirus' in class_family or 'Low_complexity' in class_family or 'LINE/L1' in class_family:
                    continue

                if 'DTA' in class_family:
                    class_family = 'DTA'
                elif 'DTC' in class_family:
                    class_family = 'DTC'
                elif 'DTM' in class_family:
                    class_family = 'DTM'
                elif 'DTT' in class_family:
                    class_family = 'DTT'
                elif 'DTH' in class_family:
                    class_family = 'DTH'
                elif 'Helitron' in class_family:
                    class_family = 'Helitron'
                elif 'hAT' in class_family:
                    class_family = 'hAT'

                records.append({
                    'repeat_name': repeat_name,
                    'class': class_family,
                    'region': region,
                    'divergence': divergence,
                    'length': length
                })
            except:
                continue

    return pd.DataFrame(records)

# === LOAD DATA ===
file_path = 'ragtag19058m.fasta.mod.panEDTA.out'
df = parse_repeatmasker_out(file_path)

# Filter regions
df = df[df['region'].isin(['SDR', 'X-NR', 'X-PAR', 'Y-PAR'])].copy()

# Bin divergence
df['div_bin'] = pd.cut(df['divergence'], bins=np.arange(0, 51, 0.5), right=False)

# Calculate total repeat length per region
region_total_bp = df.groupby('region')['length'].sum().rename('region_total')
df = df.merge(region_total_bp, on='region', how='left')

# Aggregate: sum repeat bp per class/div_bin/region
grouped = (
    df.groupby(['region', 'div_bin', 'class'])
    .agg(bp=('length', 'sum'), region_total=('region_total', 'first'))
    .reset_index()
)

# Calculate % of total repeat size
grouped['percent'] = 100 * grouped['bp'] / grouped['region_total']
grouped['div_bin'] = grouped['div_bin'].astype(str)  # For x-axis labeling

# === PLOT ===
kimura_plot_percent = (
    ggplot(grouped, aes(x='div_bin', y='percent', fill='class')) +
    geom_bar(stat='identity', position='stack') +
    facet_wrap('~region', scales='free_y') +
    labs(
        title='Kimura Divergence Plot (Y = % of Repeat Size)',
        x='Divergence from Consensus (%)',
        y='% of Total Repeat Length',
        fill='TE Class'
    ) +
    theme_classic() +
    theme(
        figure_size=(8, 6),
        axis_text_x=element_text(rotation=45, ha='right')
    )
)

# Save
kimura_plot_percent.save("kimura_percent_yaxis.pdf", dpi=300)
