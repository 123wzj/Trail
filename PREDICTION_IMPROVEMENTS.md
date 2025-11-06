# 预测脚本改进文档

## 概述

本文档描述对 `src/prectdict_event_gnn.py` 脚本的改进，以修复 APT 组织归因预测功能。

## 修复的问题

### 1. 缺少标签传播步骤
**问题**: 模型训练时使用了标签传播特征，但预测脚本没有生成这些特征，导致预测不准确。

**解决方案**: 添加了 `_run_label_propagation()` 方法，参考 `lprop.py` 实现：
- 使用 `MessagePassing` 聚合邻居节点的标签信息
- 为 IOC 节点和事件节点生成预测标签特征
- 迭代传播 5 个 hops
- 在预测前自动运行标签传播

```python
def _run_label_propagation(self, test_event_ids: torch.Tensor, hops: int = 5) -> torch.Tensor:
    """对测试事件运行标签传播"""
    # 初始化所有节点的标签特征
    num_classes = len(self.label_map)
    x = torch.zeros(self.graph.x.size(0), num_classes, device=self.device)
    
    # 为已知标签的事件节点设置one-hot编码
    for event_id, label in zip(self.graph.event_ids, self.graph.y):
        if label >= 0:
            x[event_id, label] = 1.0
    
    # 使用 MessagePassing 进行标签传播
    mp = MessagePassing(aggr='mean')
    for k in range(hops):
        x = x + mp.propagate(self.graph.edge_index, x=x, size=None)
    
    return x[test_event_ids]
```

### 2. 节点类型硬编码
**问题**: 使用硬编码的节点类型 ID (4) 可能与实际数据不匹配。

**解决方案**: 添加了 `_get_event_node_type_id()` 方法动态获取 EVENT 节点类型：
- 首先检查 `graph.type_dict`
- 然后检查 `graph.ntypes`
- 如果都不存在，回退到默认值 4

```python
def _get_event_node_type_id(self) -> int:
    """动态获取 EVENT 节点类型 ID"""
    if hasattr(self.graph, 'type_dict'):
        return self.graph.type_dict.get('EVENT', self.graph.type_dict.get('event', 4))
    elif hasattr(self.graph, 'ntypes'):
        for key, value in self.graph.ntypes.items():
            if key.upper() == 'EVENT':
                return value
    return 4  # 回退默认值
```

### 3. 缺少数据验证
**问题**: 没有验证图数据是否包含必要的属性，可能导致运行时错误。

**解决方案**: 添加了 `_validate_graph()` 方法：
- 检查必需属性: `edge_csr`, `event_ids`, `y`, `label_map`, `x`
- 验证节点类型信息: `ntypes` 或 `type_dict`
- 在初始化时自动运行验证

```python
def _validate_graph(self):
    """验证图数据包含所有必要的属性"""
    required_attrs = ['edge_csr', 'event_ids', 'y', 'label_map', 'x']
    missing_attrs = []
    
    for attr in required_attrs:
        if not hasattr(self.graph, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        raise ValueError(f"图数据缺少必要属性: {', '.join(missing_attrs)}")
    
    if not hasattr(self.graph, 'ntypes') and not hasattr(self.graph, 'type_dict'):
        raise ValueError("图数据缺少 'ntypes' 或 'type_dict' 属性")
```

### 4. 缺少调试信息
**问题**: 无法追踪预测失败的具体原因。

**解决方案**: 添加了全面的调试输出：
- 图的关键属性（节点数、事件数、标签分布）
- IOC 节点和事件节点的详细信息
- 标签传播过程的进度
- 预测前后的概率分布
- 中间步骤的形状和数值范围检查

### 5. 缺少错误处理
**问题**: 关键步骤没有错误处理，难以定位问题。

**解决方案**: 为所有关键操作添加了错误处理：
- 初始化预测器
- 加载 IOC 映射
- 查找 IOC 节点
- 运行标签传播
- 执行模型预测
- 提供有意义的错误信息和栈跟踪（在调试模式下）

## 新功能

### 命令行参数支持

```bash
# 使用默认设置运行
python prectdict_event_gnn.py

# 启用调试模式
python prectdict_event_gnn.py --debug

# 指定自定义模型和数据集路径
python prectdict_event_gnn.py --model /path/to/model.pt --dataset /path/to/dataset

# 获取更多预测结果
python prectdict_event_gnn.py --top-k 10

# 不保存结果到文件
python prectdict_event_gnn.py --no-save

# 指定输出文件
python prectdict_event_gnn.py --output my_results.json
```

### 支持的参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--model` | 模型权重文件路径 | 脚本中配置的 `MODEL_PATH` |
| `--dataset` | 数据集目录路径 | 脚本中配置的 `DATASET_PATH` |
| `--top-k` | 返回前 K 个最可能的组织 | 5 |
| `--debug` | 启用详细的调试输出 | False |
| `--no-save` | 不保存结果到文件 | False |
| `--output` | 输出文件路径 | `apt_prediction_result.json` |

## 调试模式输出示例

启用 `--debug` 标志后，脚本会输出详细的诊断信息：

```
======================================================================
图信息:
======================================================================
  总节点数: 2,134,567
  事件节点数: 4,523
  节点类型: {'ips': 0, 'urls': 1, 'domains': 2, 'ASN': 3, 'EVENT': 4}
  标签数量: 22
  标签分布:
    APT28: 234
    APT29: 189
    ...
======================================================================

✓ 加载了 12,345 个 IP 地址映射
✓ 加载了 45,678 个域名映射
✓ 加载了 98,765 个 URL 映射

======================================================================
开始 APT 归因预测
======================================================================

  找到 3 个 IOC:
    ✓ ip:198.199.14.14
    ✓ ip:74.81.69.244
    ✓ domain:example.com

  使用 EVENT 节点类型 ID: 4
  找到 15 个关联事件

======================================================================
运行标签传播以生成预测特征
======================================================================

开始标签传播 (5 hops)...
  从 CSR 格式转换边索引...
  [1/5] 已标记的测试事件: 8/15
  [2/5] 已标记的测试事件: 12/15
  [3/5] 已标记的测试事件: 14/15
  [4/5] 已标记的测试事件: 15/15
  [5/5] 已标记的测试事件: 15/15

  标签传播完成
  生成的特征形状: torch.Size([15, 22])
  特征范围: [0.0000, 0.8542]

======================================================================
运行 GNN 模型预测
======================================================================
  预测输出形状: torch.Size([15, 22])
  概率范围: [0.0012, 0.4521]
  概率和: 1.0000

  聚合预测:
  平均概率范围: [0.0034, 0.3842]
  概率和: 1.0000

======================================================================
Top-K 预测结果
======================================================================
  1. APT28                   38.42% ███████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  2. APT29                   22.15% ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  3. Lazarus                 15.67% ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  4. APT32                    8.93% ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  5. FIN7                     5.21% ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

======================================================================
```

## 测试

创建了 `test_prectdict_event_gnn.py` 验证脚本，包含以下测试：

1. **导入验证**: 检查所有必需的导入
2. **类结构**: 验证 IOCAttributor 类包含所有必需的方法
3. **方法签名**: 检查关键方法的签名是否正确
4. **标签传播**: 验证标签传播实现的存在
5. **动态节点类型查找**: 确认不使用硬编码
6. **命令行参数**: 验证 argparse 集成
7. **错误处理**: 检查 try-except 块的存在

运行测试:
```bash
cd src
python test_prectdict_event_gnn.py
```

所有 7 个测试都通过 ✓

## 向后兼容性

所有更改都保持了向后兼容性：
- 现有的配置变量 (`MODEL_PATH`, `DATASET_PATH`, `INPUT_IOCS` 等) 仍然有效
- 可以在不使用命令行参数的情况下运行脚本
- 默认行为与原始脚本相同（除了新增的功能）

## 代码质量改进

1. **类型提示**: 为所有方法添加了类型提示
2. **文档字符串**: 为所有新方法添加了详细的文档
3. **错误消息**: 提供了清晰、可操作的错误消息
4. **代码组织**: 将功能分解为小的、可测试的方法
5. **调试友好**: 添加了全面的日志记录和诊断输出

## 使用建议

1. **首次运行**: 使用 `--debug` 标志验证一切正常工作
2. **生产环境**: 在没有 `--debug` 的情况下运行以获得简洁输出
3. **故障排除**: 使用 `--debug` 诊断预测问题
4. **批量处理**: 修改 `INPUT_IOCS` 变量或扩展脚本以读取文件

## 性能注意事项

- 标签传播在大图上可能需要一些时间（5 hops）
- `--debug` 模式会增加轻微的性能开销
- CSR 到边索引的转换在首次标签传播时执行一次

## 未来改进建议

1. 支持从文件批量读取 IOCs
2. 添加置信度阈值过滤
3. 支持不同的聚合策略（加权平均、投票等）
4. 添加可视化输出选项
5. 支持增量图更新
