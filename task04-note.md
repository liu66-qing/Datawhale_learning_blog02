## 1. 

- **定义：** 机器学习子集，利用多层神经网络自动提取特征，处理复杂数据（如图像、语音）。

- **特点：** 端到端学习，从原始输入到输出，省去手动特征工程，需大量数据和计算力。

- **与传统机器学习对比：**

  | 方面       | 传统机器学习      | 深度学习               |
  | ---------- | ----------------- | ---------------------- |
  | 特征提取   | 手动设计          | 自动学习               |
  | 数据需求   | 少量可行          | 海量数据               |
  | 模型复杂度 | 浅层模型（如SVM） | 深层网络（可数百层）   |
  | 适用场景   | 结构化数据        | 非结构化数据（如图像） |

- **核心：** 表示学习（Representation Learning），通过多层变换将输入转为高层抽象表示（如像素→边缘→对象）。

------

## 2.Local Minima & Saddle Points

- **背景：** 神经网络优化通过梯度下降最小化损失函数，损失函数“地形”复杂，存在以下问题：
  - **Local Minima：** 梯度为0的非全局最优解，易导致训练停滞。
  - **Saddle Points：** 梯度≈0的平坦区域，高维空间中更常见。
- **影响：** 训练卡在局部最小值或鞍点，模型性能不佳。
- **解决方法：**
  - **随机初始化权重：** 不同起点降低卡坑概率。
  - **动量法（Momentum）：** 累积梯度方向，加速逃离鞍点。公式：
    [ v_t = \beta v_{t-1} + (1 - \beta) \nabla L, \quad \theta_t = \theta_{t-1} - \alpha v_t ] （(v_t): 速度，(\beta): 动量系数≈0.9，(\alpha): 学习率）
  - **高维空间特性：** 高维下鞍点占比高，随机梯度下降（SGD）逃离概率增加。

------

## 3. 训练技巧: Batch & Momentum

- **梯度下降变体：**
  - **批量梯度下降（Batch GD）：** 用全数据集计算梯度，稳定但慢。
  - **随机梯度下降（SGD）：** 单样本更新，快但噪声大，利于逃局部最小值。
  - **小批量（Mini-batch）：** 折中，batch size通常32~256。
- **学习率调度：**
  - 动态调整学习率，初期大（快速探索），后期小（精细优化）。
  - 策略：指数衰减 (\alpha_t = \alpha_0 \cdot e^{-\gamma t}) 或阶梯衰减。
- **其他：**
  - **Epoch vs Iteration：** Epoch=全数据集一遍，Iteration=单次参数更新。
  - **早停（Early Stopping）：** 验证集损失上升时停止，防过拟合。
- **SGD更新公式：**
  [ \theta \leftarrow \theta - \alpha \nabla L(\theta) ] （(\theta): 参数，(L): 损失）

------

## 4. Adaptive learning rate

- **问题：** 固定学习率不适应各维度梯度差异（如稀疏数据）。
- **方法：** 自适应优化器根据历史梯度调整学习率。
- **常见优化器：**
  - **Adagrad：** 累积平方梯度，公式：
    [ g_t = g_{t-1} + (\nabla L)^2, \quad \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{g_t + \epsilon}} \nabla L ] 优点：适合稀疏数据；缺点：后期学习率过小，停滞。
  - **RMSprop：** 用指数移动平均，解决Adagrad停滞问题：
    [ E[g^2]*t = \beta E[g^2]*{t-1} + (1-\beta) (\nabla L)^2 ] （(\beta \approx 0.9)）
  - **Adam：** 结合动量法和RMSprop，含偏差修正，广泛使用。

------

## 5. 分类任务损失函数

- **作用：** 衡量预测(\hat{y})与真实(y)差异，分类任务常用概率输出（Softmax）。
- **交叉熵损失（Cross-Entropy Loss）：**
  - **二分类：**
    [ L = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right] ] （(y_i \in {0,1})，(\hat{y}_i = \sigma(z)) Sigmoid输出）
  - **多分类：**
    [ L = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log \hat{y}_{i,c} ] （(C): 类别数，(\hat{y} = \text{Softmax}(z))，(z = Wx + b)）
- **Softmax函数：**
  [ \hat{y}*c = \frac{e^{z_c}}{\sum*{k=1}^C e^{z_k}} ] 确保概率和为1。
- **优势：** 交叉熵与最大似然等价，梯度稳定；相比MSE，分类任务收敛更快。

------

## 6. 归一化技巧

- **问题：** 深层网络训练不稳定，因内部协变量偏移（前层输出分布变化）。
- **批归一化（Batch Normalization）：**
  - 每层输入标准化（均值0，方差1）。
  - 步骤：
    1. 计算batch均值(\mu_B)和方差(\sigma_B^2)。
    2. 归一化：(\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}) （(\epsilon): 防除0）。
    3. 缩放平移：(y = \gamma \hat{x} + \beta) （(\gamma, \beta): 可学习参数）。
  - 位置：卷积后或全连接前。
  - 优点：加速训练，允许大初始学习率，减少调参。
- **其他归一化：**
  - **Layer Norm：** 独立于batch，适合RNN。
  - **Group Norm：** 小batch场景效果好。
- **测试时：** 用运行均值（moving average）代替batch均值。
