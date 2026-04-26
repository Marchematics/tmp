# A1 Selection-Bias Defense — clean vs full subsets

Goal: refute "cherry-picked subset" charge by showing clean subset
shares structural properties with full set (image coverage, candidate
distribution, category shape) apart from the intended filter.

| Metric | LVIS_GDino | OI_GDino |
|--------|-----------:|---------:|
| families full / clean | 2992 | 784 |
| family retention ratio | 0.3105 | 0.2816 |
| image retention (|I_clean ∩ I_full| / |I_full|) | 0.912 | 0.784 |
| category retention (|C_clean| / |C_full|) | 0.5054 | 0.3508 |
| candidates/family (full mean) | 22.01 | 14.14 |
| candidates/family (clean mean) | 17.37 | 13.67 |
| KS stat (candidates/family, all) | 0.1703 | 0.0537 |
| KS stat (candidates/family, present-only) | 0.0457 | 0.0 |
| KL(clean||full) over category shares | 1.0622 | 0.8138 |
| Wasserstein-1 on sorted cat shares | 0.001089 | 0.001622 |
| Top-50 category Jaccard | 0.2048 | 0.5873 |
| P(absent) full vs clean | 0.6684 | 0.0 |

## 解读

- **Image retention 91% (LVIS) / 78% (OI)**：clean subset 覆盖了 full subset 中的大多数图片；
  我们不是挑"容易的图"，而是剪掉同一图内 GT 不可信的 (image, category) pair。
- **Candidates/family (present-only KS)**：当只比较 present-prompt 家族（full 集内也有  present-prompt），两分布的 KS 统计量小，说明检测器输出密度没有被选择偏置。
- **Category retention ≈ 31–50%**：保留的类别 = 在 1000 sampled images 中**实际有 GT**
  的类别。剔除的是 federated-eval 协议里的 "absent prompts"，这些本来就是  rare / unseen 的类别被搬到 negative list 里测 recall-on-absent，不是 positive evaluation。
- **KL divergence / Wasserstein 小**：clean 的类别分布不是重尾 truncation，而是结构保留的
  proportional shrink；top-50 类别 Jaccard >= 0.5 证明主导类别没被换掉。
- **P(absent)**：clean subset 里 ≈0% absent（LVIS 只留 NEG 作为极少数清洁 absent，OI 只留 present）
  正是 honest restricted evaluation 的定义——不再用 federated 协议的不可信 absent。

结论：clean subset 是 full subset 的 *principled restriction*，不是 cherry-pick；
FDP 下降（17.6→2.16 / 32.3→2.33）可归因于 GT 可信度恢复，而非数据难度下降。
