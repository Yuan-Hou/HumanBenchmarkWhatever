#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版同义词编辑器
输入两个词组，自动互相添加为同义词并保存
"""

import json
import os

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

def add_synonyms(word_a, word_b, synonyms):
    """将两个词组互相添加为同义词"""
    # 检查词组是否存在
    if word_a not in synonyms:
        print(f"❌ '{word_a}' 不在同义词表中")
        return False
    
    if word_b not in synonyms:
        print(f"❌ '{word_b}' 不在同义词表中")
        return False
    
    # 添加同义词关系
    if word_b not in synonyms[word_a]:
        synonyms[word_a].append(word_b)
        print(f"✅ 已将 '{word_b}' 添加到 '{word_a}' 的同义词列表")
    
    if word_a not in synonyms[word_b]:
        synonyms[word_b].append(word_a)
        print(f"✅ 已将 '{word_a}' 添加到 '{word_b}' 的同义词列表")
    
    return True

def main():
    file_path = "hoi_synonym_dict.json"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件 {file_path} 不存在")
        return
    
    # 加载同义词字典
    synonyms = load_synonym_dict(file_path)
    if synonyms is None:
        return
    
    print(f"📖 已加载同义词字典，包含 {len(synonyms)} 个词组")
    print("🔄 输入两个词组，按回车自动添加同义词关系并保存")
    print("💡 输入 'quit' 或 'exit' 退出程序\n")
    
    while True:
        try:
            # 输入第一个词组
            word_a = input("词组 A: ").strip()
            if word_a.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not word_a:
                continue
            
            # 输入第二个词组
            word_b = input("词组 B: ").strip()
            if word_b.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not word_b:
                continue
            
            if word_a == word_b:
                print("⚠️  两个词组不能相同\n")
                continue
            
            # 添加同义词关系
            if add_synonyms(word_a, word_b, synonyms):
                # 自动保存
                if save_synonym_dict(synonyms, file_path):
                    print("💾 已保存到文件")
                else:
                    print("❌ 保存失败")
            
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误：{e}")

if __name__ == "__main__":
    main()
