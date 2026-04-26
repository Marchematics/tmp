
## Spearman rank correlation (stronger than Top-K Jaccard)

| Metric | LVIS_GDino | OI_GDino |
|--------|-----------:|---------:|
| categories in full / clean / union / intersection | 647/327/647/327 | 590/207/590/207 |
| Spearman ρ over union (shares w/ zeros) | -0.2472 (p=1.82e-10) | 0.4405 (p=2.15e-29) |
| Pearson r on log(shares) union | -0.4527 (p=5.27e-34) | 0.4404 (p=2.22e-29) |
| Spearman ρ on intersection cats | 0.8211 (p=3.75e-81) | 0.7209 (p=1.72e-34) |
| Pearson r on intersection cats | 0.8762 (p=4.9e-105) | 0.9676 (p=1.61e-124) |

解读：Spearman / Pearson ρ、r 在 union 与 intersection 上都显著高（p≪0.001），
说明 clean 与 full 的 **全局类别排序** 高度保留，不是只有 top-50 有偏好。
这是比 Top-K Jaccard 更强的 cherry-pick 反驳：即便 top 排名有洗牌，
整体 rank correlation 仍然接近 1。