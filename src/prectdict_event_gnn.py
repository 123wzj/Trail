import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import json
import argparse
from torch_geometric.nn import MessagePassing
from models.gnn import SageClassifier

# 模型和数据集路径
MODEL_PATH = '/root/PythonProject/Trail-main/model_weights/weights/2-layer/gnn_train-0.620_max_lprop+feats+ae-new-data.pt'
DATASET_PATH = '/root/PythonProject/Trail-main/src/otx_dataset'

# 输入 IOC 信息 - 在这里输入您要预测的 IOC
INPUT_IOCS = {
    'ips': [
        '198.199.14.14',
        '198.199.14.15',
        '74.81.69.244'
    ],
    'domains': [
        "origin-academy.autodesk.com",
        "partnercenterqa-q1.autodesk.com",
        "people-insights-tableau.autodesk.com"
    ],
    'urls': [
        'http://cdneu.windows8downloadscdn.com/ofr/Mamawaj/Mamawaj_V.cis',
        'http://cdnus.windows8downloadscdn.com/ofr/Mamawaj/Mamawaj_V.cis',
        "http://img.windows8downloadscdn.com/img/CH_logo.png"
    ]
}

# 返回前 K 个最可能的组织
TOP_K = 5

# 是否保存结果到文件
SAVE_RESULT = True
OUTPUT_FILE = 'apt_prediction_result.json'

class IOCAttributor:
    """基于 IOC 的 APT 组织归因预测器"""

    def __init__(self, model_path: str, dataset_path: str, debug: bool = False):
        """初始化预测器"""
        self.dataset_path = Path(dataset_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        
        # 加载模型
        sd, args, kwargs = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = SageClassifier(*args, **kwargs)
        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()

        # 加载图数据
        graph_file = self.dataset_path / 'full_graph_csr.pt'
        self.graph = torch.load(graph_file, map_location=self.device, weights_only=False)

        # 验证图数据
        self._validate_graph()
        
        # 加载 IOC 索引映射
        self._load_ioc_mappings()

        # APT 标签映射
        self.label_map = self.graph.label_map
        
        if self.debug:
            self._print_graph_info()

    def _validate_graph(self):
        """验证图数据包含所有必要的属性"""
        required_attrs = ['edge_csr', 'event_ids', 'y', 'label_map', 'x']
        missing_attrs = []
        
        for attr in required_attrs:
            if not hasattr(self.graph, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            raise ValueError(f"图数据缺少必要属性: {', '.join(missing_attrs)}")
        
        # Check for ntypes or type_dict
        if not hasattr(self.graph, 'ntypes') and not hasattr(self.graph, 'type_dict'):
            raise ValueError("图数据缺少 'ntypes' 或 'type_dict' 属性")
        
        if self.debug:
            print("✓ 图数据验证通过")
    
    def _print_graph_info(self):
        """打印图的关键属性用于调试"""
        print(f"\n{'=' * 70}")
        print("图信息:")
        print(f"{'=' * 70}")
        print(f"  总节点数: {self.graph.x.size(0):,}")
        print(f"  事件节点数: {self.graph.event_ids.size(0):,}")
        
        if hasattr(self.graph, 'ntypes'):
            print(f"  节点类型: {self.graph.ntypes}")
        elif hasattr(self.graph, 'type_dict'):
            print(f"  节点类型: {self.graph.type_dict}")
        
        # 标签分布
        unique_labels, counts = torch.unique(self.graph.y, return_counts=True)
        print(f"  标签数量: {len(self.label_map)}")
        print(f"  标签分布:")
        for label_idx, count in zip(unique_labels, counts):
            if label_idx >= 0:  # Skip -1 (unlabeled)
                apt_name = self.label_map.get(label_idx.item(), f"Unknown-{label_idx}")
                print(f"    {apt_name}: {count}")
        
        print(f"{'=' * 70}\n")
    
    def _get_event_node_type_id(self) -> int:
        """动态获取 EVENT 节点类型 ID"""
        if hasattr(self.graph, 'type_dict'):
            return self.graph.type_dict.get('EVENT', self.graph.type_dict.get('event', 4))
        elif hasattr(self.graph, 'ntypes'):
            # ntypes is a dict-like mapping
            for key, value in self.graph.ntypes.items():
                if key.upper() == 'EVENT':
                    return value
        # Fallback to 4 if not found
        return 4
    
    def _run_label_propagation(self, test_event_ids: torch.Tensor, hops: int = 5) -> torch.Tensor:
        """
        对测试事件运行标签传播
        
        参考 lprop.py 中的 prop_loop 函数实现
        使用 MessagePassing 聚合邻居节点的标签信息
        
        Args:
            test_event_ids: 需要预测的事件节点 ID
            hops: 标签传播的跳数
            
        Returns:
            传播后的标签特征 (num_events x num_classes)
        """
        if self.debug:
            print(f"\n开始标签传播 ({hops} hops)...")
        
        # 获取边索引 (如果图有 edge_index 属性)
        if not hasattr(self.graph, 'edge_index'):
            # 从 CSR 转换为 edge_index
            if self.debug:
                print("  从 CSR 格式转换边索引...")
            self.graph.edge_index = self.graph.edge_csr.to_ei()
        
        # 初始化所有节点的标签特征
        num_classes = len(self.label_map)
        x = torch.zeros(self.graph.x.size(0), num_classes, device=self.device)
        
        # 为已知标签的事件节点设置one-hot编码
        for event_id, label in zip(self.graph.event_ids, self.graph.y):
            if label >= 0:  # 跳过未标记的节点
                x[event_id, label] = 1.0
        
        # 使用 MessagePassing 进行标签传播
        mp = MessagePassing(aggr='mean')
        
        for k in range(hops):
            # 传播标签信息
            x = x + mp.propagate(self.graph.edge_index, x=x, size=None)
            
            if self.debug:
                # 统计有标签的测试事件数量
                labeled_test = (x[test_event_ids].sum(dim=1) > 0).sum()
                print(f"  [{k+1}/{hops}] 已标记的测试事件: {labeled_test}/{test_event_ids.size(0)}")
        
        # 返回测试事件的标签特征
        return x[test_event_ids]

    def _load_ioc_mappings(self):
        """加载 IOC 字符串到特征索引的映射"""
        try:
            # 加载 IPs
            ip_file = self.dataset_path / 'ips.csv'
            if ip_file.exists():
                df_ip = pd.read_csv(ip_file, sep='\t')
                self.ip_to_idx = {row['ioc']: idx for idx, row in df_ip.iterrows()}
                if self.debug:
                    print(f"✓ 加载了 {len(self.ip_to_idx)} 个 IP 地址映射")
            else:
                self.ip_to_idx = {}
                if self.debug:
                    print(f"⚠ 未找到 IP 文件: {ip_file}")

            # 加载 Domains
            domain_file = self.dataset_path / 'domains.csv'
            if domain_file.exists():
                df_domain = pd.read_csv(domain_file, sep='\t')
                self.domain_to_idx = {row['ioc']: idx for idx, row in df_domain.iterrows()}
                if self.debug:
                    print(f"✓ 加载了 {len(self.domain_to_idx)} 个域名映射")
            else:
                self.domain_to_idx = {}
                if self.debug:
                    print(f"⚠ 未找到域名文件: {domain_file}")

            # 加载 URLs
            url_file = self.dataset_path / 'urls.csv'
            if url_file.exists():
                df_url = pd.read_csv(url_file, sep='\t')
                self.url_to_idx = {row['ioc']: idx for idx, row in df_url.iterrows()}
                if self.debug:
                    print(f"✓ 加载了 {len(self.url_to_idx)} 个 URL 映射")
            else:
                self.url_to_idx = {}
                if self.debug:
                    print(f"⚠ 未找到 URL 文件: {url_file}")
        except Exception as e:
            raise RuntimeError(f"加载 IOC 映射失败: {e}")

    def _find_ioc_nodes(self, iocs: Dict[str, List[str]]):
        """在图中查找 IOC 对应的节点ID"""
        node_ids = []
        found_iocs = []
        not_found_iocs = []

        type_map = {
            'ips': (0, self.ip_to_idx),
            'domains': (2, self.domain_to_idx),
            'urls': (1, self.url_to_idx)
        }

        for ioc_type, ioc_list in iocs.items():
            if ioc_type not in type_map or not ioc_list:
                continue

            node_type_id, ioc_mapping = type_map[ioc_type]

            for ioc in ioc_list:
                feat_idx = ioc_mapping.get(ioc)
                if feat_idx is not None:
                    # 在图中查找匹配的节点
                    mask = (self.graph.x == node_type_id) & (self.graph.feat_map == feat_idx)
                    matching_nodes = mask.nonzero(as_tuple=True)[0]

                    if len(matching_nodes) > 0:
                        node_ids.extend(matching_nodes.tolist())
                        found_iocs.append(f"{ioc_type[:-1]}:{ioc}")
                    else:
                        not_found_iocs.append(f"{ioc_type[:-1]}:{ioc}")
                else:
                    not_found_iocs.append(f"{ioc_type[:-1]}:{ioc}")

        return node_ids, found_iocs, not_found_iocs

    @torch.no_grad()
    def predict_from_iocs(self, iocs: Dict[str, List[str]], top_k: int = 5):
        """基于 IOC 列表预测 APT 组织"""
        try:
            self.model.eval()
            
            if self.debug:
                print(f"\n{'=' * 70}")
                print("开始 APT 归因预测")
                print(f"{'=' * 70}")
            
            # 1. 查找 IOC 对应的节点
            node_ids, found_iocs, not_found_iocs = self._find_ioc_nodes(iocs)

            print(f"\n  找到 {len(found_iocs)} 个 IOC:")
            for ioc in found_iocs[:10]:
                print(f"    ✓ {ioc}")
            if len(found_iocs) > 10:
                print(f"    ... 还有 {len(found_iocs) - 10} 个")

            if not_found_iocs:
                print(f"\n  未找到 {len(not_found_iocs)} 个 IOC:")
                for ioc in not_found_iocs[:10]:
                    print(f"    ✗ {ioc}")
                if len(not_found_iocs) > 10:
                    print(f"    ... 还有 {len(not_found_iocs) - 10} 个")

            if not node_ids:
                print("\n错误: 没有找到任何 IOC 在图中")
                return None

            # 2. 查找这些 IOC 关联的事件节点
            # 动态获取 EVENT 节点类型 ID
            event_type_id = self._get_event_node_type_id()
            
            if self.debug:
                print(f"\n  使用 EVENT 节点类型 ID: {event_type_id}")

            # 找到与这些 IOC 相连的所有事件节点
            event_nodes = set()
            for ioc_node in node_ids:
                neighbors = self.graph.edge_csr[ioc_node]
                for neighbor in neighbors:
                    if self.graph.x[neighbor] == event_type_id:
                        event_nodes.add(neighbor.item())

            if not event_nodes:
                print("  警告: 未找到直接关联的事件节点")
                print("  使用 IOC 节点的邻居信息进行推理...")

                # 收集所有相关节点的邻居
                all_neighbors = set(node_ids)
                for nid in node_ids:
                    neighbors = self.graph.edge_csr[nid]
                    all_neighbors.update(neighbors.tolist())

                event_nodes = {n for n in all_neighbors if self.graph.x[n] == event_type_id}

                if not event_nodes:
                    print("  无法找到任何事件节点进行预测")
                    return None

            event_tensor = torch.tensor(list(event_nodes), device=self.device)
            print(f"  找到 {len(event_nodes)} 个关联事件")
            
            # 3. 运行标签传播 (关键修复!)
            if self.debug:
                print(f"\n{'=' * 70}")
                print("运行标签传播以生成预测特征")
                print(f"{'=' * 70}")
            
            # 屏蔽这些事件的标签（模拟真实推理）
            original_ids = self.graph.event_ids.clone()
            original_y = self.graph.y.clone()

            mask = ~((self.graph.event_ids == event_tensor.unsqueeze(-1)).sum(dim=0).bool())
            self.graph.event_ids = self.graph.event_ids[mask]
            self.graph.y = self.graph.y[mask]
            
            # 运行标签传播
            try:
                label_features = self._run_label_propagation(event_tensor, hops=5)
                
                if self.debug:
                    print(f"\n  标签传播完成")
                    print(f"  生成的特征形状: {label_features.shape}")
                    print(f"  特征范围: [{label_features.min():.4f}, {label_features.max():.4f}]")
            except Exception as e:
                print(f"  ⚠ 标签传播失败: {e}")
                print(f"  将继续使用模型进行预测...")

            # 4. 对事件节点进行预测
            if self.debug:
                print(f"\n{'=' * 70}")
                print("运行 GNN 模型预测")
                print(f"{'=' * 70}")

            # 前向推理
            preds = self.model.inference(self.graph, event_tensor)
            
            if self.debug:
                print(f"  预测输出形状: {preds.shape}")
                print(f"  概率范围: [{preds.min():.4f}, {preds.max():.4f}]")
                print(f"  概率和: {preds.sum(dim=1).mean():.4f}")

            # 恢复标签
            self.graph.event_ids = original_ids
            self.graph.y = original_y

            # 5. 验证模型输出
            if preds.shape[0] != len(event_nodes):
                raise ValueError(f"预测数量不匹配: 期望 {len(event_nodes)}, 得到 {preds.shape[0]}")
            
            if preds.shape[1] > len(self.label_map):
                print(f"  ⚠ 警告: 预测类别数 ({preds.shape[1]}) 大于标签映射数 ({len(self.label_map)})")
            
            # 6. 聚合多个事件的预测结果（取平均）
            avg_preds = preds.mean(dim=0)
            
            if self.debug:
                print(f"\n  聚合预测:")
                print(f"  平均概率范围: [{avg_preds.min():.4f}, {avg_preds.max():.4f}]")
                print(f"  概率和: {avg_preds.sum():.4f}")

            # 获取 top-k 预测
            top_k_probs, top_k_indices = torch.topk(avg_preds, k=min(top_k, len(self.label_map)))

            # 构建结果
            result = {
                'input_iocs': iocs,
                'found_iocs': found_iocs,
                'not_found_iocs': not_found_iocs,
                'num_events': len(event_nodes),
                'predictions': []
            }
            
            if self.debug:
                print(f"\n{'=' * 70}")
                print("Top-K 预测结果")
                print(f"{'=' * 70}")

            for rank, (prob, label_idx) in enumerate(zip(top_k_probs, top_k_indices), 1):
                apt_name = self.label_map[label_idx.item()]
                confidence = prob.item()

                result['predictions'].append({
                    'rank': rank,
                    'organization': apt_name,
                    'probability': confidence,
                    'confidence_pct': f"{confidence * 100:.2f}%"
                })

                # 打印结果
                bar_len = int(confidence * 50)
                bar = '█' * bar_len + '░' * (50 - bar_len)
                print(f"  {rank}. {apt_name:25s} {confidence * 100:6.2f}% {bar}")

            print(f"\n{'=' * 70}\n")

            return result
            
        except Exception as e:
            print(f"\n✗ 预测过程中发生错误: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    def save_result(self, result: Dict, output_file: str):
        """保存预测结果到 JSON 文件"""
        if result is None:
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {output_file}\n")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='基于 IOC 的 APT 组织归因预测器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 使用默认设置运行
  python prectdict_event_gnn.py
  
  # 启用调试模式
  python prectdict_event_gnn.py --debug
  
  # 指定自定义模型和数据集路径
  python prectdict_event_gnn.py --model /path/to/model.pt --dataset /path/to/dataset
  
  # 获取更多预测结果
  python prectdict_event_gnn.py --top-k 10
        '''
    )
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='模型权重文件路径')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH,
                        help='数据集目录路径')
    parser.add_argument('--top-k', type=int, default=TOP_K,
                        help='返回前 K 个最可能的组织')
    parser.add_argument('--debug', action='store_true',
                        help='启用详细的调试输出')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存结果到文件')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    # 验证输入 IOCs
    total_iocs = 0
    for ioc_type, ioc_list in INPUT_IOCS.items():
        if ioc_list:
            total_iocs += len(ioc_list)

    if total_iocs == 0:
        print("错误: 没有提供任何 IOC 进行预测")
        return

    # 初始化预测器
    try:
        attributor = IOCAttributor(args.model, args.dataset, debug=args.debug)
    except Exception as e:
        print(f"\n✗ 初始化预测器失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return

    # 执行预测
    try:
        result = attributor.predict_from_iocs(INPUT_IOCS, top_k=args.top_k)

        if result:
            # 保存结果
            if not args.no_save:
                attributor.save_result(result, args.output)
        else:
            print("\n✗ 预测失败: 无法生成结果")

    except Exception as e:
        print(f"\n✗ 预测执行失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
