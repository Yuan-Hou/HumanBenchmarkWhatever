import json
import os
from typing import Dict, List
import itertools
import concurrent.futures
import threading
from test_framework import QuestionGenerator
from utils import ask_question

class MultiPersonClothingFeatureQuestionGenerator(QuestionGenerator):
    """多图人体服装特征题型生成器"""
    def __init__(self, dataset_pictures):
        super().__init__(dataset_pictures)
        self.clothing_color_name_2_picture_dict: Dict[str, Dict[str, List]] = {}
        self.clothing_name_color_2_picture_dict: Dict[str, Dict[str, List]] = {}
        self.synonym_dict: Dict[str, List[str]] = {}
        self.distinguishable_dict: Dict[str, List[str]] = {}

    def _construct_clothing_dict(self, filtered_pictures):
        for pic in filtered_pictures:
            for person in pic.persons:
                clothings = person.get_clothing_list()
                for clothing in clothings:
                    name = clothing.get("name")
                    colors = clothing.get("color", [])
                    for color in colors:
                        if color not in self.clothing_color_name_2_picture_dict:
                            self.clothing_color_name_2_picture_dict[color] = {}
                        if name not in self.clothing_color_name_2_picture_dict[color]:
                            self.clothing_color_name_2_picture_dict[color][name] = []
                        self.clothing_color_name_2_picture_dict[color][name].append(pic)

                        if name not in self.clothing_name_color_2_picture_dict:
                            self.clothing_name_color_2_picture_dict[name] = {}
                        if color not in self.clothing_name_color_2_picture_dict[name]:
                            self.clothing_name_color_2_picture_dict[name][color] = []
                        self.clothing_name_color_2_picture_dict[name][color].append(pic)
        print(f"All {len(self.clothing_name_color_2_picture_dict)} clothing name-color mappings constructed: {', '.join(list(self.clothing_name_color_2_picture_dict.keys()))}")
        print(f"All {len(self.clothing_color_name_2_picture_dict)} clothing color-name mappings constructed: {', '.join(list(self.clothing_color_name_2_picture_dict.keys()))}")
        self._construct_synonym_dict(list(self.clothing_name_color_2_picture_dict.keys()), list(self.clothing_color_name_2_picture_dict.keys()))

    def _construct_synonym_dict(self, name_list, color_list):
        """构建同义词词典，使用16个并发线程，支持增量更新"""
        # 创建线程锁保护共享资源
        lock = threading.Lock()
        cnt = 0
        
        # 读取已有的同义词字典文件
        existing_synonyms = {}
        existing_distinguishable = {}
        if os.path.exists("synonym_dict.json"):
            try:
                with open("synonym_dict.json", "r") as f:
                    existing_data = json.load(f)
                    existing_synonyms = existing_data.get("synonyms", {})
                    existing_distinguishable = existing_data.get("distinguishable", {})
                print(f"Loaded existing synonym dictionary with {len(existing_synonyms)} entries.")
            except (json.JSONDecodeError, FileNotFoundError):
                print("Could not load existing synonym dictionary, starting fresh.")
        
        # 合并已有数据到当前实例
        self.synonym_dict.update(existing_synonyms)
        self.distinguishable_dict.update(existing_distinguishable)
        
        def process_name_combination(combo):
            name1, name2 = combo
            wording_overlap = False
            for word in name1.split():
                if word in name2:
                    wording_overlap = True
                    break
            for word in name2.split():
                if word in name1:
                    wording_overlap = True
                    break
            if not wording_overlap:
                return (name1, name2, False)
            ans = ask_question(f"'{name1}' and '{name2}' are words discribing two wearable items. Please analyze their meanings and decide if they are looking alike, of same meaning, or one of them belong to the other. At the end of your answer, please put 'yes' if they are some kind of synonymous as said or 'no' if they are not.")
            yes_idx = ans[::-1].lower().find("yes"[::-1])
            no_idx = ans[::-1].lower().find("no"[::-1])
            if yes_idx == -1:
                yes_idx = float('inf')
            if no_idx == -1:
                no_idx = float('inf')
            if yes_idx < no_idx:
                print(ans)
                print(yes_idx, no_idx)
            return (name1, name2, yes_idx < no_idx)
        
        def process_color_combination(combo):
            color1, color2 = combo
            wording_overlap = False
            for word in color1.split():
                if word in color2:
                    wording_overlap = True
                    break
            for word in color2.split():
                if word in color1:
                    wording_overlap = True
                    break
            if not wording_overlap:
                return (color1, color2, False)
            ans = ask_question(f"'{color1}' and '{color2}' are words discribing two color types of some wearings. Please analyze their meanings and decide if they are looking alike, of same meaning, possibly hard to distinguish, or one of them belong to the other. At the end of your answer, please put 'yes' if they are this kind of similar color pattern or 'no' if they are not.")
            print(ans)
            yes_idx = ans[::-1].lower().find("yes"[::-1])
            no_idx = ans[::-1].lower().find("no"[::-1])
            if yes_idx == -1:
                yes_idx = float('inf')
            if no_idx == -1:
                no_idx = float('inf')
            if yes_idx < no_idx:
                print(ans)
                print(yes_idx, no_idx)
            return (color1, color2, yes_idx < no_idx)
        
        def combination_already_processed(item1, item2):
            """检查组合是否已经处理过"""
            # 检查是否在同义词字典中
            if item1 in existing_synonyms and item2 in existing_synonyms:
                return True
            if item1 in existing_distinguishable and item2 in existing_distinguishable:
                return True
            return False
        
        # 初始化字典
        for name in name_list:
            if name not in self.synonym_dict:
                self.synonym_dict[name] = []
            if name not in self.distinguishable_dict:
                self.distinguishable_dict[name] = []
        
        for color in color_list:
            if color not in self.synonym_dict:
                self.synonym_dict[color] = []
            if color not in self.distinguishable_dict:
                self.distinguishable_dict[color] = []
        
        # 处理名称组合，跳过已处理的组合
        name_combinations = [combo for combo in itertools.combinations(name_list, 2) 
                           if not combination_already_processed(combo[0], combo[1])]
        total_name_combinations = len(name_combinations)
        
        print(f"Found {total_name_combinations} new name combinations to process.")
        
        if total_name_combinations > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                future_to_combo = {executor.submit(process_name_combination, combo): combo for combo in name_combinations}
                
                for future in concurrent.futures.as_completed(future_to_combo):
                    name1, name2, is_synonym = future.result()
                    
                    with lock:
                        if is_synonym:
                            self.synonym_dict[name1].append(name2)
                            self.synonym_dict[name2].append(name1)
                        else:
                            self.distinguishable_dict[name1].append(name2)
                            self.distinguishable_dict[name2].append(name1)
                        
                        cnt += 1
                        if cnt % 2000 == 0:
                            print(f"Processed {cnt}/{total_name_combinations} name combinations so far.")
                            with open("synonym_dict.json", "w") as f:
                                json.dump({
                                    "synonyms": self.synonym_dict,
                                    "distinguishable": self.distinguishable_dict
                                }, f)
        
        # 处理颜色组合，跳过已处理的组合
        color_combinations = [combo for combo in itertools.combinations(color_list, 2) 
                            if not combination_already_processed(combo[0], combo[1])]
        total_color_combinations = len(color_combinations)
        processed_colors = 0
        
        print(f"Found {total_color_combinations} new color combinations to process.")
        
        if total_color_combinations > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                future_to_combo = {executor.submit(process_color_combination, combo): combo for combo in color_combinations}
                
                for future in concurrent.futures.as_completed(future_to_combo):
                    color1, color2, is_synonym = future.result()
                    
                    with lock:
                        if is_synonym:
                            self.synonym_dict[color1].append(color2)
                            self.synonym_dict[color2].append(color1)
                        else:
                            self.distinguishable_dict[color1].append(color2)
                            self.distinguishable_dict[color2].append(color1)
                        
                        processed_colors += 1
                        if processed_colors % 100 == 0:  # 更频繁地保存，避免丢失进度
                            print(f"Processed {processed_colors}/{total_color_combinations} color combinations so far.")
                            with open("synonym_dict.json", "w") as f:
                                json.dump({
                                    "synonyms": self.synonym_dict,
                                    "distinguishable": self.distinguishable_dict
                                }, f)
        
        # 最终保存
        with open("synonym_dict.json", "w") as f:
            json.dump({
                "synonyms": self.synonym_dict,
                "distinguishable": self.distinguishable_dict
            }, f)
        
        print(f"Completed processing {total_name_combinations} new name combinations and {total_color_combinations} new color combinations.")
        print(f"Total entries in synonym dictionary: {len(self.synonym_dict)}")
        print(f"Total entries in distinguishable dictionary: {len(self.distinguishable_dict)}")
        

    def filter_pictures(self):
        """过滤符合条件的图片"""
        filtered_pictures = []
        for picture in self.dataset_pictures:
            # 需要人体占比超过30%
            body_area_sum = sum(person.body_area() for person in picture.persons)
            # 需要至少有一人有一件服饰
            has_clothing = any(len(person.get_clothing_list()) > 0 for person in picture.persons)
            if body_area_sum > 0.3 and has_clothing:
                filtered_pictures.append(picture)
        print(f"Filtered down to {len(filtered_pictures)} records for multi-person clothing feature questions.")
        return filtered_pictures
        
    def generate_questions(self):
        filtered_pictures = self.filter_pictures()
        self._construct_clothing_dict(filtered_pictures)
        # TODO: 这里可以添加更多的题目生成逻辑
        return []
