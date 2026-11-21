# Reviewer ACF 方法比较

## 结论

当前主报告使用的 `Pooled Pearson ACF` **不等同于**“对每个通道分别计算 ACF，再对所有通道的 ACF 取平均”。

`src/analysis/reviewer_acf.py` 现已同时输出所述的通道级等权统计，聚合名为 `mean_channel_pearson`，结果字段为 `mean_channel_acf`。它会写入 `electricity_training_residual_acf.csv`、`electricity_training_residual_acf.json` 和 Markdown 报告；原有主比较仍使用 `pooled_pearson`，因此不会改变既有结果或结论。

另外，当前脚本要求训练窗口数为 $N=18{,}125$，而不是 $17{,}885$；若 loader 的样本数是 $17{,}885$，脚本会在计算前停止并报错。因此，当前产出的报告不可能是基于 $17{,}885$ 个训练窗口。

## 当前主报告：Pooled Pearson ACF

对于 lag $k$，脚本收集所有满足相同 window $n$、相同通道 $d$ 的合法 pair：

$$
\mathcal{P}_k=
\left\{
\left(Y_{n,t,d},Y_{n,t+k,d}\right):
n=1,\ldots,N;\ d=1,\ldots,D;\ t=1,\ldots,H-k
\right\}.
$$

然后对全部 $N D(H-k)$ 个 pair 一次性计算 Pearson 相关系数：

$$
\operatorname{ACF}^{\mathrm{pooled}}_Y(k)=
\operatorname{Corr}\left(
\{Y_{n,t,d}\}_{(n,t,d)\in\mathcal{P}_k},
\{Y_{n,t+k,d}\}_{(n,t,d)\in\mathcal{P}_k}
\right).
$$

这里的均值、方差和协方差都是跨所有 window、通道和合法 horizon 位置汇总后计算的。窗口边界和通道边界不会产生时间 pair，但不同通道的数据会共同决定这一个全局 Pearson 系数。

## 所述方法：通道级 ACF 的等权平均

若“对每个通道分别计算 ACF”指对固定通道 $d$ 汇总该通道所有训练窗口及其合法 horizon pair，再计算 Pearson ACF，则应为：

$$
\rho_d(k)=
\operatorname{Corr}\left(
\{Y_{n,t,d}\}_{n,t},
\{Y_{n,t+k,d}\}_{n,t}
\right),
$$

$$
\operatorname{ACF}^{\mathrm{channel\ mean}}_Y(k)=
\frac{1}{|\mathcal{C}_k|}\sum_{d\in\mathcal{C}_k}\rho_d(k).
$$

其中 $\mathcal{C}_k$ 是 Pearson 相关有定义（左右侧均非恒定）的通道集合。正常 Electricity 运行中其大小应为 $D$；若有常量通道，脚本将其排除，并以 `valid_channel_count` 明确记录实际参与平均的通道数。这会让每个有效通道的 ACF 权重严格相同。它与 pooled Pearson 不同：pooled Pearson 先合并所有通道的 pair，再进行一次全局中心化和归一化；通道均值则先在每个通道内中心化和归一化，最后才平均。除非各通道在每个 lag 下具有恰好相同的统计量和相关性，两种结果通常不相等。

## 当前脚本中最接近的附加统计

脚本还输出 `mean_within_window_channel_pearson`，但它也不完全等于上述通道级方法。它计算：

$$
\frac{1}{|\mathcal{V}_k|}
\sum_{(n,d)\in\mathcal{V}_k}
\operatorname{Corr}\left(
\{Y_{n,t,d}\}_{t=1}^{H-k},
\{Y_{n,t+k,d}\}_{t=1}^{H-k}
\right),
$$

其中 $\mathcal{V}_k$ 是所有非恒定的 $(n,d)$ 序列。也就是说，它先对每一个 window--channel 序列计算 ACF，再对所有有效的 window--channel 结果等权平均；它不是“每个通道先跨所有训练窗口算一个 ACF，再对 $D$ 个通道平均”。

## 标准化的关系

Y 来自训练 loader 的 `batch_y`，已使用仅在训练时间段拟合的逐通道 StandardScaler：

$$
Y^{\mathrm{scaled}}_{t,d}=
\frac{Y^{\mathrm{raw}}_{t,d}-\mu^{\mathrm{train}}_d}
{\sigma^{\mathrm{train}}_d}.
$$

这保证不同通道在输入尺度上可比较，但不把 pooled Pearson 变成通道 ACF 的等权平均。并且，重叠 forecast window 会重复采样同一时间点，所以被汇总的 window-level Y 值不必严格满足均值为 $0$、方差为 $1$。
