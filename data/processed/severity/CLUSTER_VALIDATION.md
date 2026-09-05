# Severity Cluster Validation

This report evaluates the saved severity clustering model on the train fall-window feature matrix only.
Lower Davies-Bouldin is better; higher silhouette and Calinski-Harabasz are better.

## Validation Setup

| Item | Value |
| --- | --- |
| Train fall feature matrix | 4,484 x 58 |
| Saved production model k | 3 |
| k sweep | 2..6 |

## Saved Model Metrics

| k | silhouette | davies_bouldin | calinski_harabasz |
| --- | ---: | ---: | ---: |
| 3 | 0.1585 | 2.1518 | 778.4105 |

## Per-Cluster Silhouette for k=3

| cluster_id | count | mean_silhouette | median_silhouette | min_silhouette | max_silhouette |
| --- | --- | --- | --- | --- | --- |
| 0 | 1659 | 0.1754 | 0.1957 | -0.0382 | 0.3287 |
| 1 | 1558 | 0.1432 | 0.1440 | -0.0007 | 0.2657 |
| 2 | 1267 | 0.1551 | 0.1623 | 0.0125 | 0.2724 |

## k Sweep Summary

| k | silhouette | davies_bouldin | calinski_harabasz | silhouette_rank | davies_bouldin_rank |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.1774 | 1.9900 | 999.8137 | 1 | 3 |
| 3 | 0.1585 | 2.1518 | 778.4105 | 5 | 5 |
| 4 | 0.1643 | 2.0347 | 715.8559 | 3 | 4 |
| 5 | 0.1686 | 1.8709 | 657.6688 | 2 | 2 |
| 6 | 0.1596 | 1.8303 | 602.4927 | 4 | 1 |

## Interpretation

Best silhouette: k=2 (0.1774).
Best Davies-Bouldin: k=6 (1.8303).
Silhouette plot saved to: D:/Fall Detection/fall-detection/data/processed/severity/plots/clusters/silhouette_vs_k.png

A k=3 model is considered acceptable for exploratory severity labeling if its silhouette score is close to the best k and its Davies-Bouldin score is not materially worse than the best alternative.
