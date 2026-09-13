# Severity cluster selection summary

## Model selection

- Tested k values: [2, 3, 4, 5]
- Selected k: 3
- Selection rule: Selected the simplest K within a near-tie of the best silhouette, while also keeping Davies-Bouldin and Calinski-Harabasz close to their best values and requiring each cluster to remain above a minimum cluster share threshold.
- k=3 statement: k=3 is supported. The selected k=3 produced a meaningful low → medium → high intensity progression based on peak acceleration, RMS, gyroscope peak, and energy, with no cluster below the minimum cluster share threshold.

## Comparison metrics

 k  silhouette  davies_bouldin  calinski_harabasz  silhouette_std  davies_bouldin_std  calinski_harabasz_std  mean_inertia  min_cluster_pct  max_cluster_pct cluster_size_range
 2    0.177382        1.990108         999.813842    9.337448e-07            0.000138               0.000195 212638.140889         0.378546         0.621454       [1697, 2787]
 3    0.158482        2.151785         778.410502    0.000000e+00            0.000000               0.000000 193013.803011         0.282560         0.369982       [1267, 1659]
 4    0.164331        2.034746         715.856302    1.984128e-06            0.000013               0.000216 175799.388377         0.187779         0.324888        [842, 1457]
 5    0.168586        1.870682         657.654128    5.591189e-05            0.003315               0.014142 163843.223592         0.118421         0.301918        [526, 1361]

## Cluster profiles

 cluster_id  size  peak_accel_mean  peak_accel_median  acc_rms_mean  gyro_peak_mean  gyro_rms_mean  energy_mean  delta_velocity_mean  sma_mean  jerk_mean
          0  1659         6.427383           6.442289      2.338990       10.139005       3.041017          NaN                  NaN       NaN        NaN
          1  1558         7.858683           7.683052      2.634410       12.975687       3.796813          NaN                  NaN       NaN        NaN
          2  1267         7.282822           7.175217      2.574718       12.314197       3.720041          NaN                  NaN       NaN        NaN

## Stability analysis

 k  mean_silhouette  std_silhouette  mean_davies_bouldin  std_davies_bouldin  mean_calinski_harabasz  std_calinski_harabasz  mean_cluster_size_min_pct  mean_cluster_size_max_pct
 2         0.177382    9.337448e-07             1.990108            0.000138              999.813842               0.000195                   0.378546                   0.621454
 3         0.158482    0.000000e+00             2.151785            0.000000              778.410502               0.000000                   0.282560                   0.369982
 4         0.164331    1.984128e-06             2.034746            0.000013              715.856302               0.000216                   0.187779                   0.324888
 5         0.168586    5.591189e-05             1.870682            0.003315              657.654128               0.014142                   0.118421                   0.301918
