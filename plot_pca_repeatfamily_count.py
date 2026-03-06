### this plot pca for normalized repeat family counts for chromosomes in the genome

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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
                        region1 = 'Y-PAR'
                    elif midpoint <= 94_000_000:
                        region1 = 'SDR'
                elif chrom == 'X':
                    if midpoint > 205e6:
                        region1 = 'X-PAR'
                    elif midpoint <= 205_000_000:
                        region1 = 'X-NR'
                else:
                    region1 = chrom
                if 'pan' not in repeat_name:
                    unassignedcount[region1] += 1

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
                    region = chrom.split('r')[1] if 'r' in chrom else chrom 

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

                records.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'repeat_name': repeat_name,
                    'te_class': class_family,
                    'region': region
                })
            except Exception:
                continue

    for chrom in unassignedcount:
        print(chrom, unassignedcount[chrom])

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
    '1': 405181065,
    '2': 360744853,
    '3': 322146660,
    '4': 234605930,
    '5': 283498671,
    '6': 200167906,
    '7': 304037795,
    '8': 248427277,
    '9': 172247092
}

'''
def normalize_counts(df):
    for region in ['SDR', 'X-NR', 'X-PAR', 'Y-PAR', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        if region in df.columns:
            size_mb = region_sizes.get(region, 1_000_000) / 1_000_000
            #df[region] = df[region] / 1e6
            df[region] = df[region] / size_mb
    return df
'''
def normalize_counts(df):
    # Normalize by region size (Mb)
    for region in ['SDR', 'X-NR', 'X-PAR', 'Y-PAR', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        if region in df.columns:
            size_mb = region_sizes.get(region, 1_000_000) / 1_000_000
            df[region] = df[region] / size_mb

    return df
 



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

for col in ['SDR', 'X-NR', 'X-PAR', 'Y-PAR', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
    #for col in ['SDR', 'X-NR', 'X-PAR', 'Y-PAR', 'Autosomes']:
    if col not in summary_df.columns:
        summary_df[col] = 0

summary_df = normalize_counts(summary_df)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

region_cols = ['SDR', 'X-NR', 'X-PAR', 'Y-PAR', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#region_cols = ['SDR', 'X-NR', 'X-PAR', 'Y-PAR', 'Autosomes']
summary_df['pan_status'] = summary_df['repeat_name'].str.contains('pan', case=False).map({
    True: 'Repeat Families',
    False: 'Non-family Repeats'
})

X = summary_df[region_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
pca_df = summary_df.copy()
pca_df['PC1'] = pcs[:, 0]
pca_df['PC2'] = pcs[:, 1]

k = (
    ggplot(pca_df, aes(x='PC1', y='PC2', color='pan_status')) +
    geom_point(size=4, alpha=0.8) +
    theme_classic() +
    labs(
        title='PCA of Repeat Densities Across Multiple Regions',
        x='PC1',
        y='PC2',
        color='Pan Status'
    )
)

#print(k)
long_df = summary_df.melt(
    id_vars=['repeat_name', 'te_class'],
    value_vars=['SDR', 'X-NR', 'X-PAR', 'Y-PAR', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    #value_vars = ['SDR', 'X-NR', 'X-PAR', 'Y-PAR', 'Autosomes'],
    var_name='region',
    value_name='density'
)

#print (long_df.head())

# Pivot so rows=regions, columns=repeat_names, values=density
region_repeat_pivot = long_df.pivot_table(
    index='region',
    columns='repeat_name',
    values='density',
    #fill_value=0
)

# Now do PCA on region_repeat_pivot
scaler = StandardScaler()
X_scaled = scaler.fit_transform(region_repeat_pivot)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

# Prepare DataFrame for plotting
pca_regions_df = pd.DataFrame({
    'region': region_repeat_pivot.index,
    'PC1': pcs[:, 0],
    'PC2': pcs[:, 1]
})

# Plot regions in PCA space
k = (
    ggplot(pca_regions_df, aes(x='PC1', y='PC2', label='region')) +
    geom_point(size=5, color = 'grey', alpha = 0.3) +
    geom_text(nudge_y=0.1) +
    theme_classic() +
    labs(title='PCA of Repeat Densities: Regions as Samples', x='PC1', y='PC2')
)


long_df = summary_df.melt(
    id_vars=['repeat_name', 'pan_status'],   # pan_status is here                                                                                                                                               
    value_vars=['SDR', 'X-NR', 'X-PAR', 'Y-PAR', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    #value_vars=['SDR', 'X-NR', 'X-PAR', 'Y-PAR', 'Autosomes'],
    var_name='region',
    value_name='density'
)

# Pivot to have rows as region + pan_status, columns as repeat_name
region_pan_repeat_pivot = long_df.pivot_table(
    index=['region', 'pan_status'],
    columns='repeat_name',
    values='density',
    fill_value=0
)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(region_pan_repeat_pivot)

# PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

# Prepare dataframe for plotting
pca_df = pd.DataFrame({
    'region': [idx[0] for idx in region_pan_repeat_pivot.index],
    'pan_status': [idx[1] for idx in region_pan_repeat_pivot.index],
    'PC1': pcs[:, 0],
    'PC2': pcs[:, 1],
})

pca_df['pan_status'] = pca_df['pan_status'].astype('category')
X = pca_df[['PC1', 'PC2']].values

sil_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)
    print(f'k={k}, Silhouette Score={score:.4f}')

# Find best k
best_k = k_values[sil_scores.index(max(sil_scores))]
print(f'Best number of clusters: {best_k}')

kmeans = KMeans(n_clusters=best_k, random_state=42)
pca_df['cluster'] = kmeans.fit_predict(pca_df[['PC1', 'PC2']])

# Plot with cluster colors and ellipses
p = (
    ggplot(pca_df, aes(x='PC1', y='PC2', color = 'pan_status',  group='factor(cluster)', label='region')) +
    geom_point(size=6, alpha=0.8) +
    geom_text(nudge_y=0.1, color='black') +
    stat_ellipse(aes(group='factor(cluster)', color='factor(cluster)'), level=0.99) +
    theme_classic() +
    labs(title='PCA with Clusters and Ellipses',
         color='Cluster')
)
#print(p)



from sklearn.manifold import MDS

# Assuming region_pan_repeat_pivot is your data matrix for clustering
mds = MDS(n_components=2, random_state=42)
mds_coords = mds.fit_transform(region_pan_repeat_pivot)
mds_df = pd.DataFrame(mds_coords, columns=['MDS1', 'MDS2'])

# Add identifiers to mds_df
mds_df['region'] = region_pan_repeat_pivot.index.get_level_values('region')
mds_df['pan_status'] = region_pan_repeat_pivot.index.get_level_values('pan_status')



kmeans = KMeans(n_clusters=best_k, random_state=42)
mds_df['cluster'] = kmeans.fit_predict(mds_df[['MDS1', 'MDS2']]).astype(str)

from plotnine import *

p_mds = (
    ggplot(mds_df, aes(x='MDS1', y='MDS2', color='pan_status', label='region')) +
    geom_point(size=4, alpha=0.8) +
    geom_text(nudge_y=0.1, color='black') +
    stat_ellipse(aes(group='cluster', color='cluster'), level=0.99) +
    theme_classic() +
    labs(#title='MDS Plot with Clusters and Ellipses',
         color='Cluster')
)

#print(p_mds)



#p_mds.save("TE_Comparison_MDS.pdf", dpi=300, width=6, height=3)
#p_mds.save("TE_Comparison_MDS.svg", dpi=300, width=6, height=3)


#p_mds = (
#    ggplot(mds_df, aes(x='MDS1', y='MDS2')) +
#    # Optional: Draw minimal/invisible points for structure
#    geom_point(color='grey', alpha=0.1, size=3) +
#    
#    # Draw text labels colored by pan_status
#   geom_text(aes(label='region', color='pan_status'), size=7, fontweight='bold') +

    # Ellipses by cluster (outline only)
#    stat_ellipse(aes(group='cluster', color='cluster'), level=0.99, linetype='solid', alpha=0.6) +
#    
#    theme_classic() +
#    #labs(#title='MDS with Ellipses by Cluster and Text Colored by Pan Status',
#         #color='Pan Status / Cluster') +
#    theme(legend_position='right')
#)
#p_mds.save("TE_Comparison_MDS_v2.pdf", dpi=300, width=4, height=2)
#p_mds.save("TE_Comparison_MDS_v2.svg", dpi=300, width=4, height=2)
#print(p_mds)


###########MDSFOREACH#################################



###########PCA#####################
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from scipy.stats import chi2
from plotnine import *

def compute_ellipse(df, level=0.99, n_points=100):
    """Compute ellipse coordinates for a 2D dataset (df[['x','y']])"""
    if df.shape[0] < 3:
        return None

    cov = np.cov(df.T)
    mean = df.mean().values

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    chi2_val = np.sqrt(chi2.ppf(level, 2))
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse = (eigvecs @ np.diag(np.sqrt(eigvals)) @ circle) * chi2_val
    ellipse = ellipse.T + mean
    return pd.DataFrame(ellipse, columns=['x', 'y'])



mds_all, pca_all = [], []
mds_ellipses, pca_ellipses = [], []

for status, subset in region_pan_repeat_pivot.groupby(level='pan_status'):
    subset_data = subset.droplevel('pan_status')

    # === MDS ===
    mds = MDS(n_components=2, random_state=42)
    mds_coords = mds.fit_transform(subset_data)
    mds_df = pd.DataFrame(mds_coords, columns=['x', 'y'])
    mds_df['region'] = subset_data.index
    mds_df['pan_status'] = status

    # Cluster per group
    best_k = 2
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    mds_df['cluster'] = kmeans.fit_predict(mds_df[['x', 'y']]).astype(str)

    # Ellipse per cluster
    for clust, df_c in mds_df.groupby('cluster'):
        ell = compute_ellipse(df_c[['x', 'y']])
        if ell is not None:
            ell['cluster'] = clust
            ell['pan_status'] = status
            mds_ellipses.append(ell)

    mds_all.append(mds_df)

    # === PCA ===
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(subset_data)
    pca_df = pd.DataFrame(pca_coords, columns=['x', 'y'])
    pca_df['region'] = subset_data.index
    pca_df['pan_status'] = status

    # Cluster per group
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    pca_df['cluster'] = kmeans.fit_predict(pca_df[['x', 'y']]).astype(str)

    # Ellipse per cluster
    for clust, df_c in pca_df.groupby('cluster'):
        ell = compute_ellipse(df_c[['x', 'y']])
        if ell is not None:
            ell['cluster'] = clust
            ell['pan_status'] = status
            pca_ellipses.append(ell)

    pca_all.append(pca_df)

# Combine all
mds_df_all = pd.concat(mds_all, ignore_index=True)
pca_df_all = pd.concat(pca_all, ignore_index=True)
mds_ellipse_all = pd.concat(mds_ellipses, ignore_index=True)
pca_ellipse_all = pd.concat(pca_ellipses, ignore_index=True)



# --- MDS plot ---
p_mds = (
    ggplot(mds_df_all, aes(x='x', y='y')) +
    geom_point(color='grey', alpha=0.1, size=3) +
    geom_text(aes(label='region', color='cluster'), size=7, fontweight='bold') +
    geom_path(mds_ellipse_all, aes(x='x', y='y', group='cluster', color='cluster'), alpha=0.7, size=1) +
    facet_wrap('~pan_status', scales='free') +
    theme_classic() +
    labs(title='MDS per Pan Status (independent ellipses)',
         color='Cluster',  x = 'MDS1', y = 'MDS2') +
    theme(legend_position='right')
)

p_mds.save("TE_Comparison_MDS_faceted_precomputed_ellipses.pdf", dpi=300, width=8, height=4)
p_mds.save("TE_Comparison_MDS_faceted_precomputed_ellipses.svg", dpi=300, width=8, height=4)


# --- PCA plot ---
p_pca = (
    ggplot(pca_df_all, aes(x='x', y='y')) +
    geom_point(color='grey', alpha=0.1, size=3) +
    geom_text(aes(label='region', color='cluster'), size=7, fontweight='bold') +
    geom_path(pca_ellipse_all, aes(x='x', y='y', group='cluster', color='cluster'), alpha=0.7, size=1) +
    facet_wrap('~pan_status', scales='free') +
    theme_classic() +
    labs(title='PCA per Pan Status (independent ellipses)',
         color='Cluster', x = 'PC1', y = 'PC2') +
    theme(legend_position='right')
)

p_pca.save("TE_Comparison_PCA_faceted_precomputed_ellipses.pdf", dpi=300, width=8, height=4)
p_pca.save("TE_Comparison_PCA_faceted_precomputed_ellipses.svg", dpi=300, width=8, height=4)


