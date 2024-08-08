# fedhome：Cloud-Edge Based Personalized Federated Learning for Human Activity Recognition

### 1. 个性化数据处理

每个客户端设备（如智能手机或边缘设备）会处理其本地数据，并通过GCAE和SMOTE技术进行数据扩增，从而生成更加丰富和多样化的数据集。

#### 高维数据降维

- **GCAE的编码器（Encoder）**：首先，使用GCAE的编码器将高维的输入数据（如传感器数据、图像数据等）转换为低维的特征表示。这个过程通过多个卷积层实现，有效地减少了数据的维度，同时保留了关键的特征信息。

#### 低维特征空间中的SMOTE过采样

- **SMOTE过采样**：在低维特征空间中，应用SMOTE方法生成新的少数类样本。这些新样本通过线性插值生成，形成了一个类别平衡的数据集。这个过程显著提高了数据集的多样性和代表性。

### 2. 个性化模型训练

通过生成的类平衡数据集，每个客户端可以更有效地训练个性化模型。具体过程如下：

#### 本地模型训练

- **微调预测模型**：利用扩增后的类平衡数据集，微调本地预测网络（如多层感知器MLP）的参数，使其更适应用户的个性化数据分布。这样，模型能够更准确地预测特定用户的行为或状态。

### 3. 联邦学习框架中的个性化

在联邦学习框架下，个性化主要通过以下步骤实现：

#### 本地模型更新

- **本地训练**：每个客户端基于其本地数据和扩增数据集训练GCAE模型，并更新本地预测模型。
- **参数共享**：客户端将本地更新的模型参数发送到中央服务器。

#### 全局模型聚合

- **参数聚合**：中央服务器聚合来自多个客户端的模型参数，以形成一个全局模型。这个全局模型包含了各个客户端的本地特征，提高了模型的通用性。

#### 个性化调整

- **个性化调整**：全局模型更新后，再分发回各个客户端。客户端可以进一步基于本地数据对全局模型进行微调，从而获得更个性化的模型。

### 具体实现流程

以下是FedHome框架中实现个性化的详细流程：

1. **全局模型初始化**：
   - 中央服务器初始化GCAE模型，并将其分发给所有客户端。

2. **本地数据处理与扩增**：
   - 每个客户端使用GCAE对高维数据进行降维处理，得到低维特征表示。
   - 在低维特征空间中应用SMOTE方法生成新的少数类样本，形成类平衡数据集。

3. **本地模型训练与更新**：
   - 利用生成的类平衡数据集，微调本地预测模型的参数。
   - 客户端将更新后的模型参数上传到中央服务器。

4. **全局模型聚合**：
   - 中央服务器对所有客户端上传的模型参数进行加权平均，更新全局模型参数。

5. **个性化模型调整**：
   - 中央服务器将更新后的全局模型参数分发给各个客户端。
   - 客户端基于本地数据进一步微调全局模型参数，以实现个性化调整。

### 优缺点分析

**优点**：

- **隐私保护**：数据在本地处理，避免了隐私数据上传到中央服务器的风险。
- **数据平衡**：通过SMOTE方法在低维特征空间中生成类平衡数据集，提高了模型在少数类上的表现。
- **个性化模型**：每个客户端基于自身数据进行微调，得到更符合用户特征的个性化模型。

**缺点**：

- **计算开销**：GCAE和SMOTE的应用增加了计算复杂度，可能需要更多的计算资源。
- **通信开销**：联邦学习过程中频繁的模型参数传输增加了通信开销。
- **实现复杂性**：需要结合多种技术（GCAE、SMOTE和联邦学习），增加了实现和调试的难度。

总的来说，FedHome通过结合GCAE和SMOTE在联邦学习框架下实现了高效的数据扩增和个性化模型训练，有效解决了健康监测数据中的类别不平衡问题，并提高了模型的预测准确性和个性化能力。



# fedhome，SemiPFL，FedHAR三篇分析

https://chatgpt.com/share/509b6e16-3df3-41fb-9088-13c8dac074e5



# 初步想法：

与*Federated Clustering and Semi-Supervised Learning: A New Partnership for Personalized Human Activity Recognition*想法类似，联邦聚类加半监督，其中半监督方式采用*SemiPFL: Personalized Semi-Supervised Federated Learning Framework for Edge Intelligence*中使用**自动编码器**生成的潜在表示进行**一致性训练**，保持模型输出在相邻时间窗口的一致性。

# 解决问题：

## 1.数据异质性和标注数据不足：

不同用户设备生成的数据具有多种个性化特征和边缘异质性，导致难以开发适用于所有用户的全局模型。此外，标注这些大量数据的任务具有挑战性。

## 2.隐私保护与通信成本：

由于隐私保护的需求和通信成本的限制，集中式数据积累和训练不可行。现有的大多数方法依赖高质量的标注数据，这在边缘设备上难以实现



## 技术方案（修改中）：

在行为识别领域结合联邦学习和自动编码器进行半监督学习的流程大致如下：

### 1. 数据准备和预处理

每个客户端收集并预处理本地的传感器时序数据，包括标准化和分割成固定长度的窗口。

```
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# 每个客户端分别进行数据预处理
client_data = [preprocess_data(data) for data in all_clients_data]
```

### 2. 自动编码器设计

设计一个共享的自动编码器模型结构，包括编码器和解码器部分。

```
from keras.models import Model
from keras.layers import Input, Dense

input_dim = client_data[0].shape[1]
encoding_dim = 64

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
```

### 3. 本地模型训练

每个客户端使用本地的有标签和无标签数据训练自动编码器。

```
def train_autoencoder(client_data):
    autoencoder.fit(client_data, client_data, epochs=50, batch_size=256, shuffle=True)
    return autoencoder.get_weights()

client_weights = [train_autoencoder(data) for data in client_data]
```

#### 1. **无标签数据用于自动编码器的无监督训练**

无标签数据的主要用途是用于自动编码器的训练。自动编码器的目标是学习一个低维的特征表示，即使没有标签，它仍然可以从数据的结构中提取出有用的特征。具体来说：

- 输入：无标签数据集 `X_unlabeled`。
- 目标：输入数据本身，即训练过程中，自动编码器试图将输入数据还原（即重构输入数据）。

训练过程如下：

```
python复制代码def train_autoencoder_with_unlabeled_data(autoencoder, X_unlabeled):
    autoencoder.fit(X_unlabeled, X_unlabeled, epochs=50, batch_size=256, shuffle=True)
    return autoencoder.get_weights()
```

在这一步，自动编码器学习了如何压缩和重构无标签数据，这样它就能捕捉到数据中的重要特征。

#### 2. **有标签数据用于半监督学习**

在训练完自动编码器后，可以利用提取的特征来进一步训练分类器模型。有标签数据将用于微调（fine-tuning）编码器部分，或者在特征提取完成后直接用于分类器的训练。

- **微调编码器**：你可以选择使用有标签的数据对编码器进行微调，以便让编码器能够更好地提取与分类任务相关的特征。这个过程可以通过添加一个分类头到编码器，并利用有标签数据来训练这个组合模型。

```
def fine_tune_encoder_with_labeled_data(encoder, X_labeled, y_labeled):
    # 添加分类头
    classifier_head = Dense(num_classes, activation='softmax')(encoder.output)
    classifier_model = Model(inputs=encoder.input, outputs=classifier_head)
    classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 使用有标签数据进行微调
    classifier_model.fit(X_labeled, y_labeled, epochs=50, batch_size=256)
    return classifier_model.get_weights()
```

- **直接分类**：另一种方法是，首先用自动编码器提取无标签数据和有标签数据的特征，然后将这些特征输入到一个独立的分类器模型中。这个分类器只使用有标签数据进行训练。（**在半监督学习中，自动编码器主要利用无标签数据进行自我监督学习，目的是提取通用特征。而分类器的训练需要明确的监督信号（即标签），因此通常只使用有标签数据来优化分类器的参数。通过这种方式，分类器学习到的模型能够更好地泛化到新的有标签样本上。**）

```
def train_classifier_on_features(encoder, X_labeled, y_labeled):
    # 使用编码器提取特征
    X_labeled_features = encoder.predict(X_labeled)
    
    # 构建并训练分类器
    classifier = Sequential()
    classifier.add(Dense(32, activation='relu', input_dim=encoding_dim))
    classifier.add(Dense(num_classes, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(X_labeled_features, y_labeled, epochs=50, batch_size=256)
    
    return classifier.get_weights()
```

#### 3. **联邦学习中结合两种数据**

在联邦学习的场景下，各客户端在本地使用无标签数据进行自动编码器的训练，然后利用有标签数据进一步微调模型或训练分类器。之后，将各客户端的模型权重传输到中央服务器进行联邦聚合（例如FedAvg），形成全局模型。

### 4. 联邦聚合

中央服务器接收每个客户端的模型权重，进行联邦平均（FedAvg）以更新全局模型（***考虑是否对自动编码器进行联邦聚合，这一步存疑***）。

```
import numpy as np

def federated_averaging(client_weights):
    new_weights = list()
    for weights in zip(*client_weights):
        new_weights.append(np.mean(weights, axis=0))
    return new_weights

global_weights = federated_averaging(client_weights)
autoencoder.set_weights(global_weights)
```

### 5. 提取特征

每个客户端使用更新后的全局编码器提取本地特征。

```
encoder = Model(inputs=input_layer, outputs=encoded)
client_features = [encoder.predict(data) for data in client_data]
```

### 6. 构建和训练分类器

每个客户端使用提取的特征和本地的有标签数据训练分类器。

```
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

def train_classifier(features, labels):
    labels_cat = to_categorical(labels)
    classifier = Sequential()
    classifier.add(Dense(32, activation='relu', input_dim=encoding_dim))
    classifier.add(Dense(num_classes, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(features, labels_cat, epochs=50, batch_size=256)
    return classifier.get_weights()

client_classifiers = [train_classifier(features, labels) for features, labels in zip(client_features, client_labels)]
```

### 7. 联邦聚合分类器

中央服务器聚合各客户端的分类器权重。

```
global_classifier_weights = federated_averaging(client_classifiers)
for classifier in client_classifiers:
    classifier.set_weights(global_classifier_weights)
```

### 8. 模型评估

每个客户端使用本地测试数据评估模型性能。

```
def evaluate_classifier(classifier, features, labels):
    labels_cat = to_categorical(labels)
    return classifier.evaluate(features, labels_cat)

client_evaluations = [evaluate_classifier(classifier, features, labels) for classifier, features, labels in zip(client_classifiers, client_features, client_labels)]
```

### 总结

通过上述流程，自动编码器用于特征提取，并结合联邦学习框架实现全局模型的更新。各客户端在本地训练自动编码器并提取特征后，利用这些特征训练本地分类器。中央服务器聚合每个客户端的模型权重，形成全局模型，并将其分发回客户端以进行下一轮训练。这种方法有效结合了自动编码器的特征提取能力和联邦学习的隐私保护特点，适用于行为识别领域的传感器时序数据。

## 参考：

### 比较FedHome和SemiPFL中的自动编码器

**FedHome: Cloud-Edge Based Personalized Federated Learning for In-Home Health Monitoring**

**自动编码器作用**：

1. **生成性卷积自动编码器（GCAE）**：用于处理类别不平衡和非独立同分布（non-IID）数据。
2. **特征提取和生成**：GCAE通过卷积层编码高维健康数据，生成新的少数类别数据样本，并使用SMOTE技术在低维空间生成类别平衡的数据集。
3. **通信效率**：GCAE轻量化，减少云与边缘节点之间的通信开销。

**实现**：

- **结构**：GCAE包含卷积编码器、解码器和多层感知器（MLP）。
- **训练**：在云和边缘节点上联合训练型。
- **个性化**：用户在边缘节点上使用生成的类别平衡数据集微调模型，实现个性化健康监测。

------

**SemiPFL: Personalized Semi-Supervised Federated Learning Framework for Edge Intelligence**

**自动编码器作用**：

1. **特征提取**：通过自动编码器提取用户数据的潜在表示，生成个性化的编码器和解码器。
2. **数据一致性**：使用自动编码器生成的潜在表示进行一致性训练，保持模型输出在相邻时间窗口的一致性。
3. **个性化**：使用本地无标签数据通过一致性训练微调模型，实现个性化优化。

**实现**：

- **结构**：自动编码器包含编码器和解码器，用于提取和重构数据特征。
- **训练**：使用本地带标签和无标签数据进行微调，计算重构误差和一致性损失，更新模型参数。
- **个性化**：用户使用本地数据进一步优化自动编码器和分类器，实现个性化模型。

### 相似点

1. **隐私保护**：两者都通过联邦学习框架实现数据隐私保护，避免将用户数据上传至中央服务器。
2. **个性化**：都注重模型的个性化训练，通过本地数据微调实现个性化效果。
3. **高效通信**：两者都设计了轻量化模型，减少了通信开销，提高了效率。

### 区别

1. **目标应用**：
   - **FedHome**：专注于家庭健康监测，解决类别不平衡和非独立同分布问题。
   - **SemiPFL**：专注于边缘智能，通过一致性训练和半监督学习提高个性化模型性能。
2. **技术方法**：
   - **FedHome**：使用生成性卷积自动编码器（GCAE）生成新的少数类别样本，平衡数据集。
   - **SemiPFL**：使用自动编码器提取潜在表示，通过一致性训练保持数据一致性。
3. **个性化策略**：
   - **FedHome**：通过SMOTE在低维空间生成类别平衡数据集，然后微调模型。
   - **SemiPFL**：通过一致性训练和重构误差微调模型，实现个性化优化。





SemiPFL和fedhar中提到的一致性训练有什么区别与不同，详细解释其原理



ChatGPT

### 一致性训练原理与区别

**SemiPFL中的一致性训练**

**原理**：

- **半监督学习**：利用少量带标签数据和大量无标签数据进行训练。
- **一致性假设**：假设相邻时间窗口的数据应该有相似的模型输出。
- **无监督损失**：计算相邻时间窗口模型输出的均方误差（MSE）作为无监督损失。

**实现**：

- **数据处理**：获取相邻时间窗口的数据。
- **模型更新**：通过计算无监督梯度和监督梯度，更新模型参数。
- **个性化**：结合本地数据，进一步优化模型，适应个性化需求。

**代码示例**：

```
def compute_unsupervised_loss(model, x_t, x_t1):
    y_t = model(x_t)
    y_t1 = model(x_t1)
    loss = torch.mean((y_t - y_t1) ** 2)
    return loss
```

------

**FedHAR中的一致性训练**

**原理**：

- **在线学习**：处理实时生成的无标签数据。
- **一致性假设**：保持模型输出在相邻时间窗口的一致性。
- **无监督梯度**：计算一致性损失来生成无监督梯度。

**实现**：

- **数据处理**：获取实时生成的无标签数据。
- **模型更新**：通过一致性损失计算无监督梯度，结合监督梯度更新模型参数。
- **个性化**：使用实时数据微调模型，实现个性化优化。

**代码示例**：

```
def consistency_loss(model, x_t, x_t1):
    y_t = model(x_t)
    y_t1 = model(x_t1)
    loss = torch.mean((y_t - y_t1) ** 2)
    return loss
```

### 区别与不同

1. **数据类型**：
   - **SemiPFL**：处理半监督学习，结合标签和无标签数据。
   - **FedHAR**：处理在线学习，主要使用无标签数据。
2. **应用场景**：
   - **SemiPFL**：适用于需要半监督学习的场景，通过一致性训练提高模型性能。
   - **FedHAR**：适用于实时数据处理的场景，通过在线学习实现个性化优化。
3. **实现细节**：
   - **SemiPFL**：侧重于将标签数据和无标签数据结合，计算无监督损失。
   - **FedHAR**：侧重于实时数据的一致性训练，计算一致性损失，更新模型参数。

通过这些差异，可以看出两者在一致性训练中的具体应用和实现有所不同，但都旨在提高模型的泛化能力和个性化效果。



### SemiPFL的整体算法流程如下：

1. **初始化和数据准备**：
   - 服务器拥有用户嵌入向量列表{αj}Kj=1。
   - 每个用户有一小部分标注数据和大量未标注数据，服务器拥有来自不同分布的高分辨率数据集。
2. **生成个性化自动编码器**：
   - 服务器随机选择一个用户，并使用超网络生成该用户的个性化自动编码器模型，并将其发送给用户。
3. **用户本地更新**：
   - 用户使用其本地的标注和未标注数据对接收的自动编码器模型进行更新，并将更新后的模型发送回服务器。
4. **服务器更新超网络**：
   - 服务器计算发送和接收模型参数的差异，并使用该差异更新超网络参数和对应的用户嵌入特征。
5. **生成个性化分类器**：
   - 服务器选择与用户数据分布相似的样本，并使用用户的编码器对这些样本进行编码。
   - 服务器训练一组基础模型，并将这些模型发送给用户。用户初始化这些基础模型的权重并形成初始个性化基础模型。
6. **用户本地聚合**：
   - 用户使用其标注数据对基础模型的权重进行优化，生成最终的个性化模型。

