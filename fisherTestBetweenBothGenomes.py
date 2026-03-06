import re
from collections import defaultdict
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, geom_smooth, facet_wrap, theme_classic,
    labs, theme, guides, guide_legend, geom_text
)
from sklearn.linear_model import LinearRegression
import numpy as np

def parse_repeatmasker_out(file_path, yparBound, xParBound):
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
                repeat_name = parts[9]
                class_family = parts[10]
                midpoint = (start + end) // 2

                region = None
                if chrom == 'Y':
                    region = 'Y-PAR' if midpoint > yparBound else 'SDR'
                elif chrom == 'X':
                    region = 'X-PAR' if midpoint > xParBound else 'X-NR'
                    #print(f"chrom: {chrom}, midpoint: {midpoint} → region: {region}")
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
                #if 'pan' in repeat_name:
                records.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'repeat_name': repeat_name,
                    'te_class': class_family,
                    'region': region
                })

            except Exception as e:
                continue

    return pd.DataFrame(records)


#region_sizes = { '19058m':{
#    'SDR': 94_000_000,  
#    'X-NR': 205_000_000,
#    'X-PAR': 230e6-205e6, 
#    'Y-PAR': 117e6-94e6,  
#}, '21375m': {'SDR': 80_000_000,
#    'X-NR': 205_000_000,
#    'X-PAR': 234e6-205e6,
#    'Y-PAR': 95e6-80e6 } } 

region_sizes = {
    '19058m': {
        'SDR': 94_000_000,
        'X-NR': 205_000_000,
        'X-PAR': 230e6 - 205e6,  # 25 Mb
        'Y-PAR': 117e6 - 94e6,   # 23 Mb
    },
    '21375m': {
        'SDR': 80_000_000,
        'X-NR': 205_000_000,
        'X-PAR': 234e6 - 205e6,  # 29 Mb
        'Y-PAR': 95e6 - 80e6     # 15 Mb
    }
}


df1 = parse_repeatmasker_out('ragtag19058m.fasta.mod.panEDTA.out', 94e6, 205e6)
df1['genome'] = '19058m'
df2 = parse_repeatmasker_out('ragtag21375m.fasta.mod.panEDTA.out', 80e6, 205e6)
df2['genome'] = '21375m'


df_all = pd.concat([df1, df2], ignore_index=True)


counts = df_all.groupby(['genome', 'region', 'repeat_name']).size().reset_index(name='count')


def get_region_size(row):
    return region_sizes[row['genome']][row['region']]

counts['region_size'] = counts.apply(get_region_size, axis=1)
counts['normalized_count'] = counts['count'] / counts['region_size'] * 1e6  # copies per Mb
pivot_df = counts.pivot_table(
    index=['region', 'repeat_name'],
    columns='genome',
    values='normalized_count',
    fill_value=0
).reset_index()

pivot_df.columns.name = None
pivot_df = pivot_df.rename(columns={
    '19058m': '19058m_norm',
    '21375m': '21375m_norm'
})


from plotnine import *

plots = []
from plotnine import *
class_lookup = (
    df_all.groupby(['region', 'repeat_name'])['te_class']
    .agg(lambda x: x.value_counts().idxmax())  # most common class per repeat_name-region
    .reset_index()
)

pivot_df = pivot_df.merge(class_lookup, on=['region', 'repeat_name'], how='left')

label_data = []

for region, group in pivot_df.groupby('region'):
    X = group['19058m_norm'].values.reshape(-1, 1)
    y = group['21375m_norm'].values

    if len(X) > 1:
        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        label = f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.2f}'
        label_data.append({
            'region': region,
            'label': label,
            'x': X.max() * 0.6, 
            'y': y.max() * 0.9  
        })

label_df = pd.DataFrame(label_data)

facet_plot = (
    ggplot(pivot_df, aes(x='19058m_norm', y='21375m_norm', color='te_class')) +
    geom_point(size = 5, alpha=0.7) +
    geom_smooth(method='lm', se=True, color='black', linetype='solid') +
    geom_text(
        data=label_df,
        mapping=aes(x='x', y='y', label='label'),
        inherit_aes=False,
        size=8,
        ha='left'
    ) +
    facet_wrap('~region', scales='free') +
    labs(
        #title='Repeat Normalized Copy Number: Genome1 vs Genome2 by Region',
        x='19058m (copies per Mb)',
        y='21375m (copies per Mb)'
    ) +
    theme_classic() +
    theme(
        figure_size=(10, 5),
        axis_text_x=element_text(rotation=45, hjust=1)
    )
)

facet_plot.save("TE_Comparison_SDR_and_PARsNormalizedForGenomes_v2.pdf", dpi=300, width=10, height=6)

counts_sorted = counts.sort_values(['genome', 'region', 'count'], ascending=[True, True, False])

pivot_counts = counts.pivot_table(
    index=['region', 'repeat_name'],
    columns='genome',
    values='count',
    fill_value=0
).reset_index()


high_copy = pivot_df[
    (pivot_df['19058m_norm'] >= 0) | (pivot_df['21375m_norm'] >= 0)
]

# Step 5: Sort and print
high_copy_sorted = high_copy.sort_values(['region', '19058m_norm', '21375m_norm'], ascending=[True, False, False])
#print(high_copy_sorted.to_string(index=False))


import numpy as np

# Add a small pseudocount to avoid division by zero or log(0)
pseudocount = 1e-5

pivot_df['log2FC'] = np.log2((pivot_df['21375m_norm'] + pseudocount) / (pivot_df['19058m_norm'] + pseudocount))


#from plotnine import *

logfc_plot = (
    ggplot(pivot_df, aes(x='log2FC', fill='te_class')) +
    geom_histogram(bins=50, alpha=0.8) +
    facet_wrap('~region', scales='free_y') +
    labs(
        title='Log2 Fold Change of Repeat Abundance (21375m vs 19058m)',
        x='Log2 Fold Change (21375m / 19058m)',
        y='Number of Repeats'
    ) +
    theme_classic() +
    theme(figure_size=(12, 6))
)
logfc_plot.save("TE_log2FC_by_region.pdf", dpi=300)


import pandas as pd
from scipy.stats import fisher_exact

# Make sure we have TE class info
counts_with_class = counts.merge(
    class_lookup, on=['region', 'repeat_name'], how='left'
)

# Aggregate counts per region × TE class × genome
agg = (
    counts_with_class
    .groupby(['region', 'te_class', 'genome'])['count']
    .sum()
    .reset_index()
)

# Pivot so both genome counts are on same row
agg_pivot = (
    agg.pivot_table(
        index=['region', 'te_class'],
        columns='genome',
        values='count',
        fill_value=0
    )
    .reset_index()
)

results = []

for _, row in agg_pivot.iterrows():

    region = row['region']
    te_class = row['te_class']

    # Extract counts
    a = row.get('19058m', 0)     # TE class count in genome 19058m
    b = row.get('21375m', 0)     # TE class count in genome 21375m

    # Build 2×2 contingency table:
    # [ TE class counts ]
    # [ All other TEs in same region ]
    other = (
        agg_pivot[
            (agg_pivot['region'] == region) &
            (agg_pivot['te_class'] != te_class)
        ]
    )

    # Sum counts of all other TE classes
    c = other['19058m'].sum() if '19058m' in agg_pivot.columns else 0
    d = other['21375m'].sum() if '21375m' in agg_pivot.columns else 0

    table = [[a, b],
             [c, d]]

    # Fisher test
    oddsratio, pval = fisher_exact(table, alternative='two-sided')

    results.append({
        'region': region,
        'te_class': te_class,
        'count_19058m': a,
        'count_21375m': b,
        'other_19058m': c,
        'other_21375m': d,
        'odds_ratio': oddsratio,
        'p_value': pval
    })

fisher_df = pd.DataFrame(results)
fisher_df.to_csv("fisher_test_results.tsv", sep="\t", index=False)
