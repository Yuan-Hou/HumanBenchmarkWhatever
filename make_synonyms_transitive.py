#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同义词传递闭包脚本
让同义词的同义词也是同义词，形成全连接的同义词子图
"""

import json
import os
from collections import defaultdict, deque

def load_synonym_dict(file_path):
    """加载同义词字典"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['synonyms']
    except Exception as e:
        print(f"加载文件失败：{e}")
        return None

def save_synonym_dict(synonyms, file_path):
    """保存同义词字典"""
    data = {"synonyms": synonyms}
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存文件失败：{e}")
        return False

def build_graph(synonyms):
    """构建同义词图"""
    graph = defaultdict(set)
    
    for word, synonym_list in synonyms.items():
        # 添加自己到图中
        graph[word] = set(synonym_list)
        
        # 添加反向连接
        for synonym in synonym_list:
            if synonym in synonyms:  # 确保同义词也在字典中
                graph[synonym].add(word)
    
    return graph

def find_connected_components(graph):
    """使用BFS找到所有连通分量"""
    visited = set()
    components = []
    
    for word in graph:
        if word not in visited:
            # BFS找连通分量
            component = set()
            queue = deque([word])
            visited.add(word)
            component.add(word)
            
            while queue:
                current = queue.popleft()
                
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            
            if len(component) > 1:  # 只关心有多个词的连通分量
                components.append(component)
    
    return components

def make_fully_connected(synonyms, components):
    """让每个连通分量内的所有词全连接"""
    changes_made = 0
    
    for component in components:
        component_list = list(component)
        
        # 对于连通分量中的每个词
        for word in component_list:
            if word in synonyms:
                # 获取当前同义词列表
                current_synonyms = set(synonyms[word])
                
                # 应该包含的同义词（连通分量中的所有其他词）
                should_have = component - {word}
                
                # 找到需要添加的同义词
                to_add = should_have - current_synonyms
                
                if to_add:
                    synonyms[word].extend(list(to_add))
                    changes_made += len(to_add)
                    print(f"为 '{word}' 添加了 {len(to_add)} 个同义词: {', '.join(sorted(to_add))}")
    
    return changes_made

def print_statistics(synonyms):
    """打印统计信息"""
    total_words = len(synonyms)
    words_with_synonyms = sum(1 for word_list in synonyms.values() if word_list)
    total_synonym_relations = sum(len(word_list) for word_list in synonyms.values())
    
    print(f"\n📊 统计信息：")
    print(f"总词组数: {total_words}")
    print(f"有同义词的词组数: {words_with_synonyms}")
    print(f"同义词关系总数: {total_synonym_relations}")
    
    # 找到最大的同义词组
    max_synonyms = 0
    max_word = ""
    for word, synonym_list in synonyms.items():
        if len(synonym_list) > max_synonyms:
            max_synonyms = len(synonym_list)
            max_word = word
    
    if max_synonyms > 0:
        print(f"最多同义词的词组: '{max_word}' ({max_synonyms} 个同义词)")

def main():
    file_path = "hoi_synonym_dict.json"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件 {file_path} 不存在")
        return
    
    # 加载同义词字典
    print("📖 正在加载同义词字典...")
    synonyms = load_synonym_dict(file_path)
    if synonyms is None:
        return
    
    print(f"✅ 已加载同义词字典，包含 {len(synonyms)} 个词组")
    
    # 打印初始统计信息
    print_statistics(synonyms)
    
    print("\n🔗 正在构建同义词图...")
    # 构建图
    graph = build_graph(synonyms)
    
    print("🔍 正在查找连通分量...")
    # 找到连通分量
    components = find_connected_components(graph)
    
    print(f"📈 找到 {len(components)} 个需要全连接的同义词组")
    
    if not components:
        print("✅ 所有同义词组已经是全连接的，无需修改")
        return
    
    # 显示将要处理的同义词组
    print("\n🔄 将要处理的同义词组：")
    for i, component in enumerate(components, 1):
        print(f"  组 {i}: {', '.join(sorted(component))} ({len(component)} 个词)")
    
    print(f"\n⚡ 正在生成传递闭包...")
    # 让每个连通分量全连接
    changes_made = make_fully_connected(synonyms, components)
    
    if changes_made > 0:
        print(f"\n✅ 共添加了 {changes_made} 个同义词关系")
        
        # 打印更新后的统计信息
        print_statistics(synonyms)
        
        # 保存文件
        print("\n💾 正在保存文件...")
        if save_synonym_dict(synonyms, file_path):
            print("✅ 已保存到文件")
        else:
            print("❌ 保存失败")
    else:
        print("✅ 没有需要添加的同义词关系")

if __name__ == "__main__":
    main()
