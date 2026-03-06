### this takes pan edta file 
### normalize repeat family counts by size of chromosomes
### compare normalized counts for regions and fisher test to check the differences



import re
from collections import defaultdict
import pandas as pd
from plotnine import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score

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
                repeat_name = parts[9]
                class_family = parts[10]
                midpoint = (start + end) // 2
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
                elif 'Simple_repeat' in class_family:
                    class_family  = 'Simple_repeat'
                else:
                    continue
                fam = class_family
                #if class_family not in ['Ty3/Gypsy', 'Ty1/Copia', 'LTR/unknown']:
                records.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'repeat_name': repeat_name,
                    'te_class': fam,
                    'region': region
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

# Helper function to calculate regression equation labels per facet
def get_regression_labels(df):
    labels = []
    comparisons = df['comparison'].unique()

    for comp in comparisons:
        sub = df[df['comparison'] == comp]
        X = sub['x'].values.reshape(-1, 1)
        y = sub['y'].values
        if len(X) < 2:
            # Not enough points to fit a line
            labels.append({'comparison': comp, 'x': 0, 'y': 0, 'label': 'n/a'})
            continue
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(X, y)

        label = f'y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r2:.2f}'
        # Place label near the max x and y values, adjust a bit
        labels.append({
            'comparison': comp,
            'x': sub['x'].max() * 0.2,
            'y': sub['y'].max() * 0.9,
            'label': label
        })

    return pd.DataFrame(labels)

file_path = 'ragtag19058m.fasta.mod.panEDTA.out'  # <-- change to your file path
df = parse_repeatmasker_out(file_path)
df['midpoint'] = (df['start'] + df['end']) // 2
df['bin'] = (df['midpoint'] // 1_000_000) * 1_000_000

binned_counts = (
    df[df['region'].isin(['SDR', 'X-NR', 'X-PAR', 'Y-PAR'])]
    .groupby(['region', 'bin'])
    .size()
    .reset_index(name='repeat_count')
)


summary_df = get_counts(df)

for col in ['SDR', 'X-NR', 'X-PAR', 'Y-PAR']:
    if col not in summary_df.columns:
        summary_df[col] = 0

summary_df = normalize_counts(summary_df)

sdr_xnr_df = summary_df[['repeat_name', 'te_class', 'SDR', 'X-NR']].copy()
sdr_xnr_df['x'] = sdr_xnr_df['X-NR']
sdr_xnr_df['y'] = sdr_xnr_df['SDR']
sdr_xnr_df['comparison'] = 'SDR vs X-NR'

# Y-PAR vs X-PAR
ypar_xpar_df = summary_df[['repeat_name', 'te_class', 'Y-PAR', 'X-PAR']].copy()
ypar_xpar_df['x'] = ypar_xpar_df['X-PAR']
ypar_xpar_df['y'] = ypar_xpar_df['Y-PAR']
ypar_xpar_df['comparison'] = 'Y-PAR vs X-PAR'



def get_regression_labels(df):
    from sklearn.linear_model import LinearRegression
    import numpy as np

    labels = []
    grouped = df.groupby(['pan_status', 'comparison'])

    for (pan_status, comparison), group in grouped:
        x = group['x'].values.reshape(-1, 1)
        y = group['y'].values

        if len(x) < 2:
            continue  

        model = LinearRegression().fit(x, y)
        r_squared = model.score(x, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        label = f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.2f}'


        #x_pos = group['x'].min()
        x_pos = 60-25
        #y_pos = group['y'].max() - 3
        y_pos = 60-3

        labels.append({
            'x': x_pos,
            'y': y_pos,
            'label': label,
            'comparison': comparison,
            'pan_status': pan_status
        })

    return pd.DataFrame(labels)



combined_df = pd.concat([sdr_xnr_df, ypar_xpar_df], ignore_index=True)
combined_df['pan_status'] = combined_df['repeat_name'].str.contains('pan', case=False).map({True: 'Repeat Families', False: 'Non-family Repeats'})
regression_labels = get_regression_labels(combined_df)

p = (
    ggplot(combined_df, aes(x='x', y='y', color='te_class')) +
    geom_point(size=5, alpha=0.8) +
    geom_smooth(method='lm', se=True, color='black', linetype='solid') +
    geom_text(
    data=regression_labels,
    mapping=aes(x='x', y='y', label='label'),
    inherit_aes=False,
    size=10,
    ha='left'
    )
     +
    #facet_grid('pan_status ~ comparison', scales='free') +
    facet_grid('pan_status ~ comparison') + 
    theme_classic(base_size = 14) +
    guides(color=guide_legend(ncol=1)) +
    labs(
        #title='TE Family Repeat Density Comparison',
        x='Repeat Density on X (copies/Mb)',
        y='Repeat Density on Y (copies/Mb)',
        color='TE Class'
    ) +
    theme(
        legend_position='right',
        figure_size=(8, 4) 
    ) + xlim(0, 60)
)

combined_df['facet_var'] = combined_df['te_class']
print (p)
p.save("TE_Comparison_SDR_and_PARsNormalized_v2.pdf", dpi=300, width=8, height=4)
p.save("TE_Comparison_SDR_and_PARsNormalized_v2.svg", dpi=300, width=8, height=4)
import numpy as np
import pandas as pd

def regression_labels_per_facet(df, x_col='x', y_col='y', group_cols=['pan_status', 'facet_var']):
    labels = []
    for _, group in df.groupby(group_cols):
        x = group[x_col].values
        y = group[y_col].values
        if len(x) < 2:  
            continue
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        r2 = r2_score(y, y_pred)
        label = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.2f}"
        # place label at max x
        labels.append({
            x_col: x.max()/2,
            #y_col: y_pred.max(),
            y_col: 60-3,
            'label': label,
            'pan_status': group['pan_status'].iloc[0],
            'facet_var': group['facet_var'].iloc[0]
        })
    return pd.DataFrame(labels)


regression_labels = regression_labels_per_facet(combined_df)


p = (
    ggplot(combined_df, aes(x='x', y='y', color='te_class')) +
    geom_point(size=8, alpha=0.8) +
    geom_smooth(method='lm', se=True, color='black', linetype='solid') +
    geom_text(
        data=regression_labels,
        mapping=aes(x='x', y='y', label='label'),
        inherit_aes=False,
        size=12,
        ha='left'
    ) +
    #facet_grid('pan_status ~ facet_var', scales='free') +
    facet_grid('pan_status ~ facet_var') +
    theme_classic(base_size=12) +
    guides(color=guide_legend(ncol=1)) +
    labs(
        x='Repeat Density on X (copies/Mb)',
        y='Repeat Density on Y (copies/Mb)',
        color='TE Class'
    ) +
    theme(
        legend_position='right',
        #figure_size=(25, 15) 
    ) 
)

#print (p)
#p.save("NormalizedCopyCountNR_TE.pdf", dpi=300, width=15, height=10)
#p.save("NormalizedCopyCountNR_TE.svg", dpi=300, width=15, height=10)

p.save("NormalizedCopyCountNR_Other.pdf", dpi=300, width=18, height=15)
p.save("NormalizedCopyCountNR_Other.svg", dpi=300, width=18, height=15)

'''
from scipy.stats import fisher_exact, chi2_contingency
import pandas as pd

results = []

# Loop through each comparison (e.g., "SDR vs X-NR", "Ypar vs Xpar", etc.)
for comparison, group in combined_df.groupby('comparison'):
    contingency = pd.crosstab(group['pan_status'], group['te_class'])
    
    # Decide which test to use based on table shape
    if contingency.shape == (2, 2):
        test = 'Fisher'
        oddsratio, p = fisher_exact(contingency)
        results.append({
            'comparison': comparison,
            'test': test,
            'odds_ratio': oddsratio,
            'p_value': p
        })
    else:
        test = 'Chi-squared'
        chi2, p, dof, expected = chi2_contingency(contingency)
        results.append({
            'comparison': comparison,
            'test': test,
            'chi2': chi2,
            'dof': dof,
            'p_value': p
        })

fisher_results = pd.DataFrame(results)
print(fisher_results)
'''

from scipy.stats import fisher_exact
import pandas as pd

results = []

# Loop through each comparison
'''
for comparison, group in combined_df.groupby('comparison'):
    # Loop through each TE class in this comparison
    for te_class in group['te_class'].unique():
        # Count presence/absence in each pan_status
        repeat_with = ((group['pan_status'] == 'Repeat Families') & (group['te_class'] == te_class)).sum()
        repeat_without = ((group['pan_status'] == 'Repeat Families') & (group['te_class'] != te_class)).sum()
        nonrepeat_with = ((group['pan_status'] == 'Non Repeat Families') & (group['te_class'] == te_class)).sum()
        nonrepeat_without = ((group['pan_status'] == 'Non Repeat Families') & (group['te_class'] != te_class)).sum()

        # Build 2x2 table
        table = [[repeat_with, repeat_without],
                 [nonrepeat_with, nonrepeat_without]]

        # Skip completely empty tables
        if sum(map(sum, table)) == 0:
            continue

        # Run Fisher's exact test
        oddsratio, p = fisher_exact(table, alternative='two-sided')
        results.append({
            'comparison': comparison,
            'te_class': te_class,
            'repeat_with': repeat_with,
            'nonrepeat_with': nonrepeat_with,
            'odds_ratio': oddsratio,
            'p_value': p
        })

fisher_results = pd.DataFrame(results)
print(fisher_results)
'''

from scipy.stats import fisher_exact
import pandas as pd

results = []

# filter only SDR vs X-NR
df_sdr_xnr = combined_df[combined_df['comparison'] == 'SDR vs X-NR']

for pan_status in df_sdr_xnr['pan_status'].unique():
    df_pan = df_sdr_xnr[df_sdr_xnr['pan_status'] == pan_status]
    
    for te_class in df_pan['te_class'].unique():
        # counts for this TE class
        this_te = df_pan[df_pan['te_class'] == te_class]
        a = this_te['y'].sum()  # SDR
        b = this_te['x'].sum()  # X-NR
        
        # counts for all other TE classes in the same pan_status
        other_te = df_pan[df_pan['te_class'] != te_class]
        c = other_te['y'].sum()  # SDR
        d = other_te['x'].sum()  # X-NR
        
        table = [[a, b],
                 [c, d]]
        
        # skip empty tables
        if sum(map(sum, table)) == 0:
            continue
        
        oddsratio, p_value = fisher_exact(table, alternative='two-sided')
        
        results.append({
            'pan_status': pan_status,
            'te_class': te_class,
            'SDR_counts': a,
            'XNR_counts': b,
            'odds_ratio': oddsratio,
            'p_value': p_value
        })

fisher_results = pd.DataFrame(results)
print(fisher_results)
fisher_results.to_csv("fisher_results_NRs.tsv", sep='\t', index=False)

from scipy.stats import fisher_exact
import pandas as pd

results2 = []

# filter only SDR vs X-NR

df_par = combined_df[combined_df['comparison'] == 'Y-PAR vs X-PAR']
for pan_status in df_par['pan_status'].unique():
    df_pan = df_par[df_par['pan_status'] == pan_status]

    for te_class in df_pan['te_class'].unique():
        this_te = df_pan[df_pan['te_class'] == te_class]
        a = this_te['y'].sum() 
        b = this_te['x'].sum() 

        other_te = df_pan[df_pan['te_class'] != te_class]
        c = other_te['y'].sum()
        
        d = other_te['x'].sum()
        

        table = [[a, b],
                 [c, d]]

        if sum(map(sum, table)) == 0:
            continue

        oddsratio, p_value = fisher_exact(table, alternative='two-sided')

        results2.append({
            'pan_status': pan_status,
            'te_class': te_class,
            'Y-PAR_counts': a,
            'X-PAR_counts': b,
            'odds_ratio': oddsratio,
            'p_value': p_value
        })




fisher_results2 = pd.DataFrame(results2)
print(fisher_results2)
fisher_results2.to_csv("fisher_results_PARs.tsv", sep='\t', index=False)
