import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict
import json
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

    def __init__(self, model_path: str, dataset_path: str):
        """初始化预测器"""
        self.dataset_path = Path(dataset_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载模型
        sd, args, kwargs = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = SageClassifier(*args, **kwargs)
        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()

        # 加载图数据
        graph_file = self.dataset_path / 'full_graph_csr.pt'
        self.graph = torch.load(graph_file, map_location=self.device, weights_only=False)

        # 加载 IOC 索引映射
        self._load_ioc_mappings()

        # APT 标签映射
        self.label_map = self.graph.label_map

    def _load_ioc_mappings(self):
        """加载 IOC 字符串到特征索引的映射"""
        # 加载 IPs
        ip_file = self.dataset_path / 'ips.csv'
        if ip_file.exists():
            df_ip = pd.read_csv(ip_file, sep='\t')
            self.ip_to_idx = {row['ioc']: idx for idx, row in df_ip.iterrows()}
        else:
            self.ip_to_idx = {}

        # 加载 Domains
        domain_file = self.dataset_path / 'domains.csv'
        if domain_file.exists():
            df_domain = pd.read_csv(domain_file, sep='\t')
            self.domain_to_idx = {row['ioc']: idx for idx, row in df_domain.iterrows()}
        else:
            self.domain_to_idx = {}

        # 加载 URLs
        url_file = self.dataset_path / 'urls.csv'
        if url_file.exists():
            df_url = pd.read_csv(url_file, sep='\t')
            self.url_to_idx = {row['ioc']: idx for idx, row in df_url.iterrows()}
        else:
            self.url_to_idx = {}

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
        self.model.eval()
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

        # 找到与这些 IOC 相连的所有事件节点
        event_nodes = set()
        for ioc_node in node_ids:
            neighbors = self.graph.edge_csr[ioc_node]
            for neighbor in neighbors:
                if self.graph.x[neighbor] == 4:  # 4 = EVENT
                    event_nodes.add(neighbor.item())

        if not event_nodes:
            print("  警告: 未找到直接关联的事件节点")
            print("  使用 IOC 节点的邻居信息进行推理...")

            # 收集所有相关节点的邻居
            all_neighbors = set(node_ids)
            for nid in node_ids:
                neighbors = self.graph.edge_csr[nid]
                all_neighbors.update(neighbors.tolist())

            event_nodes = {n for n in all_neighbors if self.graph.x[n] == 4}

            if not event_nodes:
                print(" 无法找到任何事件节点进行预测")
                return None

        event_tensor = torch.tensor(list(event_nodes), device=self.device)
        print(f"  找到 {len(event_nodes)} 个关联事件")

        # 3. 对事件节点进行预测

        # 屏蔽这些事件的标签（模拟真实推理）
        original_ids = self.graph.event_ids.clone()
        original_y = self.graph.y.clone()

        mask = ~((self.graph.event_ids == event_tensor.unsqueeze(-1)).sum(dim=0).bool())
        self.graph.event_ids = self.graph.event_ids[mask]
        self.graph.y = self.graph.y[mask]

        # 前向推理
        preds = self.model.inference(self.graph, event_tensor)

        # 恢复标签
        self.graph.event_ids = original_ids
        self.graph.y = original_y

        # 4. 聚合多个事件的预测结果（取平均）
        avg_preds = preds.mean(dim=0)

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

    def save_result(self, result: Dict, output_file: str):
        """保存预测结果到 JSON 文件"""
        if result is None:
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {output_file}\n")


def main():
    total_iocs = 0
    for ioc_type, ioc_list in INPUT_IOCS.items():
        if ioc_list:
            total_iocs += len(ioc_list)

    if total_iocs == 0:
        return


    # 初始化预测器
    try:
        attributor = IOCAttributor(MODEL_PATH, DATASET_PATH)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return

    # 执行预测
    try:
        result = attributor.predict_from_iocs(INPUT_IOCS, top_k=TOP_K)

        if result:
            # 保存结果
            if SAVE_RESULT:
                attributor.save_result(result, OUTPUT_FILE)
        else:
            print("\n✗ 预测失败: 无法生成结果")

    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
