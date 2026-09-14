# Severity cluster selection summary

## Model selection

- Tested k values: [2, 3, 4, 5]
- Selected k: 2
- Selection rule: Selected the simplest K within a near-tie of the best silhouette, while also keeping Davies-Bouldin and Calinski-Harabasz close to their best values and requiring each cluster to remain above a minimum cluster share threshold.
- k=3 statement: k=3 is not supported. k=3 was not selected because the clustering metrics and cluster profiles did not show a clearly defensible three-level low/medium/high progression, or the cluster-size distribution was too imbalanced for a stable three-class severity map.

## Comparison metrics

 k  silhouette  davies_bouldin  calinski_harabasz  silhouette_std  davies_bouldin_std  calinski_harabasz_std  mean_inertia  min_cluster_pct  max_cluster_pct cluster_size_range
 2    0.177382        1.990108         999.813842    9.337448e-07            0.000138               0.000195 212638.140889         0.378546         0.621454       [1697, 2787]
 3    0.158482        2.151785         778.410502    0.000000e+00            0.000000               0.000000 193013.803011         0.282560         0.369982       [1267, 1659]
 4    0.164331        2.034746         715.856302    1.984128e-06            0.000013               0.000216 175799.388377         0.187779         0.324888        [842, 1457]
 5    0.168586        1.870682         657.654128    5.591189e-05            0.003315               0.014142 163843.223592         0.118421         0.301918        [526, 1361]

## Cluster profiles

 cluster_id  size  peak_accel_mean  peak_accel_median  acc_rms_mean  gyro_peak_mean  gyro_rms_mean  energy_mean  delta_velocity_mean  sma_mean  jerk_mean
          0  1697         6.396673           6.429793      2.337363       10.155057       3.044335          NaN                  NaN       NaN        NaN
          1  2787         7.635105           7.504840      2.612292       12.703870       3.770196          NaN                  NaN       NaN        NaN

## Stability analysis

 k  mean_silhouette  std_silhouette  mean_davies_bouldin  std_davies_bouldin  mean_calinski_harabasz  std_calinski_harabasz  mean_cluster_size_min_pct  mean_cluster_size_max_pct
 2         0.177382    9.337448e-07             1.990108            0.000138              999.813842               0.000195                   0.378546                   0.621454
 3         0.158482    0.000000e+00             2.151785            0.000000              778.410502               0.000000                   0.282560                   0.369982
 4         0.164331    1.984128e-06             2.034746            0.000013              715.856302               0.000216                   0.187779                   0.324888
 5         0.168586    5.591189e-05             1.870682            0.003315              657.654128               0.014142                   0.118421                   0.301918
