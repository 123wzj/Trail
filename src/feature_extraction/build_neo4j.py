import json
import glob
import os
from tqdm import tqdm
from neo4j import GraphDatabase
from urllib.parse import urlparse

class TrailNeo4jBuilder:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="neo4j123"):
        """初始化 Neo4j 连接"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f" 已连接到 Neo4j: {uri}")

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def clear_database(self):
        """清空数据库"""
        with self.driver.session() as session:
            print("清空现有数据...")
            session.run("MATCH (n) DETACH DELETE n")
            print("数据库已清空")

    def create_constraints(self):
        """创建唯一性约束（与项目 schema 一致）"""
        print("\n创建约束和索引...")

        constraints = [
            # 事件 ID 唯一
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:EVENT) REQUIRE e.id IS UNIQUE",

            # IOC value 唯一
            "CREATE CONSTRAINT ip_value IF NOT EXISTS FOR (i:IP) REQUIRE i.value IS UNIQUE",
            "CREATE CONSTRAINT domain_value IF NOT EXISTS FOR (d:domain) REQUIRE d.value IS UNIQUE",
            "CREATE CONSTRAINT url_value IF NOT EXISTS FOR (u:URL) REQUIRE u.value IS UNIQUE",
            "CREATE CONSTRAINT asn_value IF NOT EXISTS FOR (a:ASN) REQUIRE a.value IS UNIQUE",
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"{constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"创建约束失败: {e}")

    def import_single_event(self, event_file):
        """导入单个 JSON 事件文件"""
        with open(event_file, 'r') as f:
            event_data = json.load(f)

        event_id = event_data['event_id']
        apt_label = event_data['label']
        source = event_data.get('source', 'OTX')
        details = event_data.get('details', {})

        with self.driver.session() as session:
            # 1. 创建事件节点
            event_query = """
            MERGE (e:EVENT {id: $event_id})
            SET e.label = $label,
                e.source = $source,
                e.name = $name,
                e.description = $description,
                e.tags = $tags
            RETURN e
            """

            session.run(
                event_query,
                event_id=event_id,
                label=apt_label,
                source=source,
                name=details.get('name', ''),
                description=details.get('description', ''),
                tags=details.get('tags', [])
            )

            # 2. 处理每个 IOC
            for ioc_data in event_data.get('iocs', []):
                ioc_value = ioc_data.get('ioc')
                ioc_type = ioc_data.get('type', '').upper()

                if not ioc_value:
                    continue

                try:
                    if ioc_type in ['IP', 'IPV4', 'IPV6']:
                        self._process_ip(session, event_id, ioc_data)

                    elif ioc_type in ['DOMAIN', 'HOSTNAME']:
                        self._process_domain(session, event_id, ioc_data)

                    elif ioc_type == 'URL':
                        self._process_url(session, event_id, ioc_data)

                except Exception as e:
                    print(f"处理 IOC 失败: {ioc_value} ({ioc_type}) - {e}")

    def _process_ip(self, session, event_id, ioc_data):
        """处理 IP 类型的 IOC"""
        ip_value = ioc_data['ioc']

        # 创建 IP 节点
        ip_query = """
        MERGE (i:IP {value: $value})
        SET i.country_code = $country_code,
            i.latitude = $latitude,
            i.longitude = $longitude,
            i.city = $city,
            i.region = $region
        """

        session.run(
            ip_query,
            value=ip_value,
            country_code=ioc_data.get('country_code'),
            latitude=ioc_data.get('latitude'),
            longitude=ioc_data.get('longitude'),
            city=ioc_data.get('city'),
            region=ioc_data.get('region')
        )

        # 创建事件到 IP 的关系
        event_ip_query = """
        MATCH (e:EVENT {id: $event_id}), (i:IP {value: $ip_value})
        MERGE (e)-[:CONTAINS]->(i)
        """
        session.run(event_ip_query, event_id=event_id, ip_value=ip_value)

        # 处理 ASN
        asn_str = ioc_data.get('asn')
        if asn_str:
            # 解析 ASN 字符串（格式: "AS15169 Google LLC"）
            parts = asn_str.split(' ', 1)
            asn_number = parts[0]  # "AS15169"
            issuer = parts[1] if len(parts) > 1 else ''  # "Google LLC"

            # 创建 ASN 节点
            asn_query = """
            MERGE (a:ASN {value: $asn_number})
            SET a.issuer = $issuer
            """
            session.run(asn_query, asn_number=asn_number, issuer=issuer)

            # 创建 IP 到 ASN 的关系
            ip_asn_query = """
            MATCH (i:IP {value: $ip_value}), (a:ASN {value: $asn_number})
            MERGE (i)-[:IN_ASN_GROUP]->(a)
            """
            session.run(ip_asn_query, ip_value=ip_value, asn_number=asn_number)

        # 处理 DNS 解析记录（IP -> domain）
        for resolution in ioc_data.get('resolves_to', []):
            hostname = resolution.get('host')
            if hostname:
                # 创建域名节点
                domain_query = """
                MERGE (d:domain {value: $domain_value})
                """
                session.run(domain_query, domain_value=hostname)

                # 创建 IP 到 domain 的关系
                ip_domain_query = """
                MATCH (i:IP {value: $ip_value}), (d:domain {value: $domain_value})
                MERGE (i)-[:RESOLVES_TO]->(d)
                """
                session.run(ip_domain_query, ip_value=ip_value, domain_value=hostname)

    def _process_domain(self, session, event_id, ioc_data):
        """处理 domain 类型的 IOC"""
        domain_value = ioc_data['ioc']

        # 创建 domain 节点
        domain_query = """
        MERGE (d:domain {value: $value})
        SET d.first_seen = $first_seen,
            d.last_seen = $last_seen,
            d.has_nxdomain = $has_nxdomain
        """

        # 处理 DNS 记录
        dns_records = ioc_data.get('dns_records', [])
        has_nxdomain = any(
            record.get('address') == 'NXDOMAIN'
            for record in dns_records
        )

        # 获取第一次和最后一次见到的时间
        first_seen = None
        last_seen = None
        if dns_records:
            try:
                from dateutil.parser import parse
                timestamps = []
                for record in dns_records:
                    if record.get('first'):
                        timestamps.append(parse(record['first']).isoformat())
                    if record.get('last'):
                        timestamps.append(parse(record['last']).isoformat())

                if timestamps:
                    first_seen = min(timestamps)
                    last_seen = max(timestamps)
            except:
                pass

        session.run(
            domain_query,
            value=domain_value,
            first_seen=first_seen,
            last_seen=last_seen,
            has_nxdomain=has_nxdomain
        )

        # 创建事件到 domain 的关系（间接，通过 IP 或 URL）
        # 注意：neo_to_features.py 不直接查询 EVENT->domain
        # 所以这里不创建直接关系

        # 为 DNS 记录创建关系（domain -> IP）
        for record in dns_records:
            address = record.get('address')
            record_type = record.get('record_type')

            # A 和 AAAA 记录指向 IP
            if record_type in ['A', 'AAAA'] and address and address != 'NXDOMAIN':
                # 创建 IP 节点
                ip_query = """
                MERGE (i:IP {value: $ip_value})
                """
                session.run(ip_query, ip_value=address)

                # 创建 domain 到 IP 的关系（双向）
                domain_ip_query = """
                MATCH (d:domain {value: $domain_value}), (i:IP {value: $ip_value})
                MERGE (d)-[:RESOLVES_TO]->(i)
                MERGE (i)-[:RESOLVES_TO]->(d)
                """
                session.run(domain_ip_query, domain_value=domain_value, ip_value=address)

    def _process_url(self, session, event_id, ioc_data):
        """处理 URL 类型的 IOC"""
        url_value = ioc_data['ioc']
        hostname = ioc_data.get('hostname') or ioc_data.get('net_loc')

        # 如果没有 hostname，尝试从 URL 解析
        if not hostname:
            try:
                hostname = urlparse(url_value).netloc
            except:
                pass

        # 创建 URL 节点
        url_query = """
        MERGE (u:URL {value: $value})
        SET u.hostname = $hostname,
            u.server = $server,
            u.http_code = $http_code,
            u.filetype = $filetype,
            u.fileclass = $fileclass,
            u.encoding = $encoding,
            u.expiration = $expiration
        """

        session.run(
            url_query,
            value=url_value,
            hostname=hostname,
            server=ioc_data.get('server'),
            http_code=ioc_data.get('http_code'),
            filetype=ioc_data.get('filetype'),
            fileclass=ioc_data.get('fileclass'),
            encoding=ioc_data.get('encoding'),
            expiration=ioc_data.get('expiration') or ioc_data.get('expires')
        )

        # 创建事件到 URL 的关系
        event_url_query = """
        MATCH (e:EVENT {id: $event_id}), (u:URL {value: $url_value})
        MERGE (e)-[:CONTAINS]->(u)
        """
        session.run(event_url_query, event_id=event_id, url_value=url_value)

        # 创建 URL 到 domain 的关系
        if hostname:
            # 创建 domain 节点
            domain_query = """
            MERGE (d:domain {value: $domain_value})
            """
            session.run(domain_query, domain_value=hostname)

            # 创建 URL 到 domain 的关系（双向）
            url_domain_query = """
            MATCH (u:URL {value: $url_value}), (d:domain {value: $domain_value})
            MERGE (u)-[:HOSTED_ON]->(d)
            MERGE (d)-[:HOSTS]->(u)
            """
            session.run(url_domain_query, url_value=url_value, domain_value=hostname)

        # 创建 URL 到 IP 的关系（如果有）
        if ioc_data.get('ip'):
            ip_value = ioc_data['ip']

            # 创建 IP 节点
            ip_query = """
            MERGE (i:IP {value: $ip_value})
            """
            session.run(ip_query, ip_value=ip_value)

            # 创建 URL 到 IP 的关系（双向）
            url_ip_query = """
            MATCH (u:URL {value: $url_value}), (i:IP {value: $ip_value})
            MERGE (u)-[:RESOLVES_TO]->(i)
            MERGE (i)-[:RESOLVES_TO]->(u)
            """
            session.run(url_ip_query, url_value=url_value, ip_value=ip_value)

    def import_from_directory(self, data_dir):
        """从目录批量导入 JSON 文件"""
        print(f"\n开始从 {data_dir} 导入数据...")

        # 获取所有 APT 目录
        apt_dirs = glob.glob(os.path.join(data_dir, '*'))
        apt_dirs = [d for d in apt_dirs if os.path.isdir(d)]

        total_events = 0
        total_errors = 0

        for apt_dir in apt_dirs:
            apt_name = os.path.basename(apt_dir)
            json_files = glob.glob(os.path.join(apt_dir, '*.json'))

            if not json_files:
                continue

            print(f"\n处理 {apt_name}: {len(json_files)} 个事件")

            for json_file in tqdm(json_files, desc=apt_name):
                try:
                    self.import_single_event(json_file)
                    total_events += 1
                except Exception as e:
                    total_errors += 1
                    print(f"导入失败: {os.path.basename(json_file)} - {e}")

        print(f"\n总共导入 {total_events} 个事件")
        if total_errors > 0:
            print(f"失败 {total_errors} 个事件")

    def get_statistics(self):
        """获取图数据库统计信息"""
        print("\n图数据库统计:")

        queries = {
            'EVENT': "MATCH (e:EVENT) RETURN count(e) as count",
            'IP': "MATCH (i:IP) RETURN count(i) as count",
            'domain': "MATCH (d:domain) RETURN count(d) as count",
            'URL': "MATCH (u:URL) RETURN count(u) as count",
            'ASN': "MATCH (a:ASN) RETURN count(a) as count",
            'CONTAINS': "MATCH ()-[r:CONTAINS]->() RETURN count(r) as count",
            'IN_ASN_GROUP': "MATCH ()-[r:IN_ASN_GROUP]->() RETURN count(r) as count",
            'RESOLVES_TO': "MATCH ()-[r:RESOLVES_TO]->() RETURN count(r) as count",
            'HOSTED_ON': "MATCH ()-[r:HOSTED_ON]->() RETURN count(r) as count",
        }

        with self.driver.session() as session:
            for name, query in queries.items():
                result = session.run(query)
                count = result.single()['count']
                print(f"  {name:20s}: {count:,}")

    def verify_schema(self):
        """验证数据库 schema 是否正确"""
        print("\n验证 Neo4j Schema...")

        checks = [
            ("EVENT 节点有 label 属性", "MATCH (e:EVENT) WHERE e.label IS NOT NULL RETURN count(e) as count"),
            ("IP 节点有 value 属性", "MATCH (i:IP) WHERE i.value IS NOT NULL RETURN count(i) as count"),
            ("ASN 节点有 issuer 属性", "MATCH (a:ASN) WHERE a.issuer IS NOT NULL RETURN count(a) as count"),
            ("EVENT->IP 关系存在", "MATCH (:EVENT)-[:CONTAINS]->(:IP) RETURN count(*) as count"),
            ("IP->ASN 关系存在", "MATCH (:IP)-[:IN_ASN_GROUP]->(:ASN) RETURN count(*) as count"),
        ]

        with self.driver.session() as session:
            all_passed = True
            for check_name, query in checks:
                result = session.run(query)
                count = result.single()['count']
                if count == 0:
                    all_passed = False

            if all_passed:
                print("\nSchema 验证通过！")
            else:
                print("\n部分检查未通过，请检查数据")


def main():
    # 配置
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'neo4j123')

    # JSON 数据目录 - 从 pull.py 和 enrich.py 生成的
    DATA_DIR = "/root/PythonProject/Trail-main/src/dataset_output/"

    # 初始化构建器
    builder = TrailNeo4jBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        builder.clear_database()

        # 创建约束
        builder.create_constraints()

        # 导入数据
        builder.import_from_directory(DATA_DIR)

        # 显示统计
        builder.get_statistics()

        # 验证 schema
        builder.verify_schema()

        print("\n知识图谱构建完成！")
        print("\n下一步操作:")
        print("   1. 运行: python build_graph_from_neo4j.py")
        print("   2. 然后运行: python train_gnn.py --dataset otx_dataset")

    finally:
        builder.close()


if __name__ == '__main__':
    main()
