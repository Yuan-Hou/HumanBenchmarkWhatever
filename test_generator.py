import json
import os
from copy import deepcopy
import pickle
from typing import Dict, List
import numpy as np
import csv
from dotenv import load_dotenv

from functools import cache
import itertools
from rich.progress import track
import concurrent.futures
import threading

from utils import ask_question

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH", "./final_labeling")

FACE_ATTR_NAMES = ['5 oClock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes', 'Bald', 'Bangs', 'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair', 'Bushy Eyebrows', 'Chubby', 'Double Chin', 'Eyeglasses', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones', 'Male', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard', 'Oval Face', 'Pale Skin', 'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks', 'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair', 'Wearing Earrings', 'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace', 'Wearing Necktie', 'Young']
FACE_ATTR_ADMIT_THRESHOLD = np.array([0.80, 0.70, 0.80, 0.50, 0.20, 0.90,                                   0.70,       0.95,       0.50,        0.50,         0.70,     0.70,         0.80,             0.95,      0.80,         0.92,         0.50,      0.50,        0.75,           0.80,              0.98,   0.995,                0.30,        0.75,         0.98,       0.70,        0.40,        0.40,           0.60,                0.14,         0.60,        0.80,      0.70,            0.70,        0.70,                0.75,         0.60,               0.50,               0.70,              0.98])
FACE_ATTR_DENY_THRESHOLD =  np.array([0.10, 0.05, 0.02, 0.05, 0.01, 0.05,                                   0.05,       0.05,       0.005,       0.01,         0.05,     0.01,         0.30,             0.40,      0.10,         0.005,        0.02,      0.005,       0.003,          0.01,              0.002,  0.02,                 0.005,       0.08,         0.40,       0.08,        0.004,       0.02,           0.01,                0.001,        0.02,        0.04,      0.05,            0.05,        0.02,                0.001,        0.01,               0.01,               0.001,             0.50])


_full_data = None

class HoiObject:
    def __init__(self, data):
        self.raw_data = data
        self.box = data.get("box", None)

class Hoi:
    def __init__(self, data, obj: HoiObject):
        self.raw_data = data
        self.obj = obj

class Person:
    def __init__(self, data, detect_results):
        self.raw_data = data
        self.hois = []
        if data.get("without_face") is not True and data.get("face_box") is not None:
            self.face_box = detect_results["face_boxes"][data.get("face_box")]
        else:
            self.face_box = None

        if data.get("body_box") is not None:
            self.body_box = detect_results["body_boxes"][data.get("body_box")]
        else:
            self.body_box = None

        if data.get("skeleton") is not None:
            self.skeleton = detect_results["skeletons"][data.get("skeleton")]
        else:
            self.skeleton = None

    def init_hoi_objects(self, objs: list[HoiObject]):
        for hoi in self.raw_data.get("hoi", []):
            if "no interaction" in hoi["relationship"]["action"]:
                continue
            if hoi.get("deleted") is not True and objs[hoi.get("object")] is not None:
                self.hois.append(Hoi(hoi, objs[hoi.get("object")]))

    def get_face_box(self):
        return self.raw_data.get("face_box", None)
    
    def detailing_property(self, key, default=None):
        return self.raw_data.get("qwen_detailing", {}).get(key, default)

    @cache
    def face_area(self):
        """计算人脸区域占整张图片的比例，整张图片大小为1"""
        if self.face_box is not None:
            return (self.face_box[3] - self.face_box[1]) * (self.face_box[2] - self.face_box[0])
        return 0
    
    @cache
    def body_area(self):
        """计算身体区域占整张图片的比例，整张图片大小为1"""
        if self.body_box is not None:
            return (self.body_box[3] - self.body_box[1]) * (self.body_box[2] - self.body_box[0])
        return 0

    def get_face_attr_vec(self, attr_names = None):
        if self.raw_data.get("facex_detailing"):
            if attr_names is not None:
                return np.array([self.raw_data["facex_detailing"]["attributes"].get(name, 0) for name in attr_names])
            return np.array([i for i in self.raw_data["facex_detailing"]["attributes"].values()])
        return None

    @cache
    def get_face_attr_admit_list(self):
        if self.raw_data.get("facex_detailing"):
            ans = []
            feat_vec = self.get_face_attr_vec()
            admit_vec = feat_vec >= FACE_ATTR_ADMIT_THRESHOLD
            ans = [name for name, admitted in zip(FACE_ATTR_NAMES, admit_vec) if admitted]
            return frozenset(ans)
        return frozenset()
    
    @cache
    def get_face_attr_deny_list(self):
        if self.raw_data.get("facex_detailing"):
            ans = []
            feat_vec = self.get_face_attr_vec()
            deny_vec = feat_vec < FACE_ATTR_ADMIT_THRESHOLD
            ans = [name for name, denied in zip(FACE_ATTR_NAMES, deny_vec) if denied]
            return frozenset(ans)
        return frozenset()

    def get_face_attr_assert_belief(self, admit_set, deny_set):
        if self.raw_data.get("facex_detailing"):
            result = 1.0
            for admit in admit_set:
                result *= self.raw_data["facex_detailing"]["attributes"].get(admit, 0)
            for deny in deny_set:
                result *= (1 - self.raw_data["facex_detailing"]["attributes"].get(deny, 0))
            return result
        return 0

    def get_clothing_list(self):
        return self.raw_data.get("qwen_detailing", {}).get("clothing", [])


class Picture:
    def __init__(self, data):
        self.raw_data = data
        self.persons = [Person(p, data["detect_results"]) for p in data.get("persons", []) if p.get("deleted") is not True]
        self.hoi_objects = []
        for obj in data.get("objects", []):
            if obj.get("deleted") is not True:
                self.hoi_objects.append(HoiObject(obj))
            else:
                self.hoi_objects.append(None)
        for person in self.persons:
            person.init_hoi_objects(self.hoi_objects)

    def image_path(self):
        return os.path.join(DATASET_PATH, self.raw_data.get("image_path").split("/")[-1])

def get_full_data():
    global _full_data
    if _full_data is not None:
        return _full_data
    if os.path.exists(os.path.join(DATASET_PATH, "full_data.pkl")):
        with open(os.path.join(DATASET_PATH, "full_data.pkl"), "rb") as f:
            _full_data = pickle.load(f)
            return deepcopy(_full_data)
    data = []
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".json"):
            with open(os.path.join(DATASET_PATH, filename), "r") as f:
                file_data = json.load(f)
                data.append(file_data)
    _full_data = data
    if not os.path.exists(os.path.join(DATASET_PATH, "full_data.pkl")):
        with open(os.path.join(DATASET_PATH, "full_data.pkl"), "wb") as f:
            pickle.dump(_full_data, f)
    return deepcopy(_full_data)

# ================== 题型生成器基类 ==================

class QuestionGenerator:
    """题目生成器基类"""
    
    def __init__(self, dataset_pictures):
        self.dataset_pictures = dataset_pictures
        self.picture_occurrence = {}
    
    def filter_pictures(self):
        """过滤图片，子类需要重写此方法"""
        raise NotImplementedError("Subclasses must implement filter_pictures method")
    
    def generate_questions(self):
        """生成题目，子类需要重写此方法"""
        raise NotImplementedError("Subclasses must implement generate_questions method")
    
    def save_questions(self, questions, filename):
        """保存题目到文件"""
        with open(filename, "w") as f:
            json.dump(questions, f, indent=4)
        print(f"Generated {len(questions)} questions and saved to {filename}")

# ================= 多图人体服装特征题型生成器 ================

class MultiPersonClothingFeatureQuestionGenerator(QuestionGenerator):
    """多图人体服装特征题型生成器"""
    def __init__(self, dataset_pictures):
        super().__init__(dataset_pictures)
        self.clothing_color_name_2_picture_dict: Dict[str, Dict[str, List[Picture]]] = {}
        self.clothing_name_color_2_picture_dict: Dict[str, Dict[str, List[Picture]]] = {}
        self.synonym_dict: Dict[str, List[str]] = {}
        self.distinguishable_dict: Dict[str, List[str]] = {}
        # self.various_color = ["colorful", "various", "multicolor", "various colors", "multicolored", "multi", "multi-colored", "various (seems to have beads of different colors)", "multi-colored", "multi-color", "various colors from the graphic", "multi-colored pattern", "and black", ]
        # self.no_color = ["none", "unknown", ""]

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
                if word in name2.split():
                    wording_overlap = True
                    break
            for word in name2.split():
                if word in name1.split():
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
            for word in name1.split():
                if word in name2.split():
                    wording_overlap = True
                    break
            for word in name2.split():
                if word in name1.split():
                    wording_overlap = True
                    break
            if not wording_overlap:
                return (name1, name2, False)
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


# ================== 多图人脸特征题型生成器 ==================

class MultiFaceFeatureQuestionGenerator(QuestionGenerator):
    """多图人脸特征题型生成器"""
    
    def filter_pictures(self):
        """过滤符合条件的图片"""
        # 找出其中单张面部框占屏幕比例超过3%的、没有其他foreground的人脸、face_seen为true、人没有被删除、不是no_face
        filtered_pictures = []
        for picture in self.dataset_pictures:
            # 需要有占比超过3%的面部框
            if not any(person.face_area() > 0.03 for person in picture.persons):
                continue
            # 占比低于3%的面部框里面，不能有face_seen并且不是background的
            if any(person.face_area() < 0.03 and (person.detailing_property("face_seen", True) and not person.detailing_property("background", False)) for person in picture.persons):
                continue
            filtered_pictures.append(picture)
        
        print(f"Filtered down to {len(filtered_pictures)} records after applying face feature criteria.")
        return filtered_pictures
    
    def _process_attribute_combinations(self, filtered_pictures):
        """处理人脸属性组合"""
        combine_domains = {}
        cnt = 0
        
        # 遍历FACE_ATTR_NAMES中任意三个特征组成的三元组，添加rich进度条
        for combo in track(itertools.combinations(FACE_ATTR_NAMES, 3), description="Processing face attribute combinations..."):
            # 过滤符合条件的图片:图中存在一人符合三元组
            fullfit_filtered = self._find_fullfit_pictures(filtered_pictures, combo)
            
            if len(fullfit_filtered) == 0:
                continue

            # 过滤符合条件的图片:图中存在一人（a）仅符合三元组中任意两个条件
            duo_filtered = self._find_duo_pictures(filtered_pictures, combo)
            
            # 过滤符合条件的图片：图中存在一人（a）仅符合三元组中任意一个条件
            solo_filtered = self._find_solo_pictures(filtered_pictures, combo)
            
            # 过滤符合条件的图片：图中所有人都否定了所有的属性
            none_filtered = self._find_none_pictures(filtered_pictures, combo)

            if len(fullfit_filtered) + len(duo_filtered) + len(solo_filtered) + len(none_filtered) == 0:
                continue

            combine_domains[frozenset(combo)] = {
                "fullfit": fullfit_filtered,
                "duo": duo_filtered,
                "solo": solo_filtered,
                "none": none_filtered
            }
            cnt += 1
            if cnt % 1000 == 0:
                print(f"Processed {cnt} combinations so far.")
        
        return combine_domains
    
    def _find_fullfit_pictures(self, filtered_pictures, combo):
        """找出完全符合三元组属性的图片"""
        fullfit_filtered = []
        for picture in filtered_pictures:
            found = False
            for person in picture.persons:
                # 只考虑有面部框且面部区域占比超过3%的人
                if person.face_box is not None and person.face_area() > 0.03:
                    # 检查该人是否同时具备三元组中的所有属性
                    has_all_attrs = True
                    for attr in combo:
                        if attr not in person.get_face_attr_admit_list():
                            has_all_attrs = False
                            break
                    if has_all_attrs:
                        found = True
                        break
            if found:
                fullfit_filtered.append(picture)
                self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
        return fullfit_filtered
    
    def _find_duo_pictures(self, filtered_pictures, combo):
        """找出符合两个属性、否定一个属性的图片"""
        duo_filtered = []
        for picture in filtered_pictures:
            found = False
            for person in picture.persons:
                if person.face_box is not None and person.face_area() > 0.03:
                    # 检查该人是否同时具备三元组中的任意两个属性，并确定要否定的属性
                    admit_count = 0
                    admit_subset = set()
                    deny_attr = None
                    for attr in combo:
                        if attr in person.get_face_attr_admit_list():
                            admit_count += 1
                            admit_subset.add(attr)
                        elif attr in person.get_face_attr_deny_list():
                            deny_attr = attr
                    if admit_count == 2 and deny_attr is not None:
                        # 检查其他人是否都否定了deny_attr
                        if all(deny_attr not in other_person.get_face_attr_admit_list() 
                               for other_person in picture.persons 
                               if (other_person != person and other_person.face_box is not None and other_person.face_area() > 0.03)):
                            found = True
                            break
            if found:
                duo_filtered.append((picture, admit_subset, set([deny_attr])))
                self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
        return duo_filtered
    
    def _find_solo_pictures(self, filtered_pictures, combo):
        """找出符合一个属性、否定两个属性的图片"""
        solo_filtered = []
        for picture in filtered_pictures:
            found = False
            for person in picture.persons:
                if person.face_box is not None and person.face_area() > 0.03:
                    # 检查该人是否同时具备三元组中的任意一个属性，并确定要否定的属性
                    admit_count = 0
                    admit_attr = None
                    deny_attrs = set()
                    for attr in combo:
                        if attr in person.get_face_attr_admit_list():
                            admit_count += 1
                            admit_attr = attr
                        elif attr in person.get_face_attr_deny_list():
                            deny_attrs.add(attr)
                    if admit_count == 1 and (len(deny_attrs) == 2):
                        # 检查其他人是否都否定了deny_attr
                        if all(other_person.get_face_attr_deny_list().issuperset(deny_attrs)
                               for other_person in picture.persons 
                               if (other_person != person and other_person.face_box is not None and other_person.face_area() > 0.03)):
                            found = True
                            break
            if found:
                solo_filtered.append((picture, set([admit_attr]), deny_attrs))
                self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
        return solo_filtered
    
    def _find_none_pictures(self, filtered_pictures, combo):
        """找出所有人都否定所有属性的图片"""
        none_filtered = []
        deny_attrs = set(combo)
        for picture in filtered_pictures:
            if all(other_person.get_face_attr_deny_list().issuperset(deny_attrs)
                   for other_person in picture.persons
                   if (other_person.face_box is not None and other_person.face_area() > 0.03)):
                none_filtered.append(picture)
                self.picture_occurrence[picture] = self.picture_occurrence.get(picture, 0) + 1
        return none_filtered
    
    def _calculate_penalty(self, **kwargs):
        """计算图片的惩罚值"""
        confidence = 0
        picture = kwargs["picture"]
        admit_attrs = kwargs.get("admit_attrs", set())
        deny_attrs = kwargs.get("deny_attrs", set())
        for person in picture.persons:
            person_confidence = person.get_face_attr_assert_belief(admit_attrs, deny_attrs)
            if person_confidence > confidence:
                confidence = person_confidence
        occurrence = self.picture_occurrence.get(picture, 0)
        return occurrence * (1 - confidence)
    
    def generate_questions(self):
        """生成多图人脸特征题目"""
        filtered_pictures = self.filter_pictures()
        combine_domains = self._process_attribute_combinations(filtered_pictures)
        
        # 取得出题用的数据，准备往模板里填充
        questions = []
        cnt = 0
        for combine, domain in combine_domains.items():
            fullfit_pictures = domain["fullfit"]
            duo_pictures = domain["duo"]
            solo_pictures = domain["solo"]
            none_pictures = domain["none"]
            
            if len(fullfit_pictures) > 10:
                fullfit_pictures = sorted(fullfit_pictures, key=lambda pic: self._calculate_penalty(picture=pic, admit_attrs=combine), reverse=False)[:10]
            if len(duo_pictures) > len(fullfit_pictures):
                duo_pictures = sorted(duo_pictures, key=lambda item: self._calculate_penalty(picture=item[0], admit_attrs=item[1], deny_attrs=item[2]), reverse=False)[:len(fullfit_pictures)]
            else:
                duo_pictures = duo_pictures * (len(fullfit_pictures) // len(duo_pictures) + 1)
                duo_pictures = duo_pictures[:len(fullfit_pictures)]
            if len(solo_pictures) > 10:
                solo_pictures = sorted(solo_pictures, key=lambda item: self._calculate_penalty(picture=item[0], admit_attrs=item[1], deny_attrs=item[2]), reverse=False)[:10]
            else:
                solo_pictures = solo_pictures * (len(fullfit_pictures) // len(solo_pictures) + 1)
                solo_pictures = solo_pictures[:len(fullfit_pictures)]
            if len(none_pictures) > 10:
                none_pictures = sorted(none_pictures, key=lambda pic: self._calculate_penalty(picture=pic, deny_attrs=combine), reverse=False)[:10]
            else:
                none_pictures = none_pictures * (len(fullfit_pictures) // len(none_pictures) + 1)
                none_pictures = none_pictures[:len(fullfit_pictures)]

            for fullfit_pic, duo_pic, solo_pic, none_pic in zip(fullfit_pictures, duo_pictures, solo_pictures, none_pictures):
                question = {
                    "type": "multi_face_feature",
                    "combine": list(combine),
                    "fullfit": fullfit_pic.image_path(),
                    "duo": duo_pic[0].image_path(),
                    "duo_admit": list(duo_pic[1]),
                    "solo": solo_pic[0].image_path(),
                    "solo_admit": list(solo_pic[1]),
                    "none": none_pic.image_path()
                }
                questions.append(question)
            cnt += 1
            if cnt % 2000 == 0:
                with open("questions_partial.json", "w") as f:
                    json.dump(questions, f)
                print(f"Generated {len(questions)} questions so far.")
        
        return questions

if __name__ == "__main__":
    full_data = get_full_data()
    dataset_pictures = [Picture(p) for p in full_data]
    print(f"Loaded {len(full_data)} records from dataset.")

    multi_clothing_generator = MultiPersonClothingFeatureQuestionGenerator(dataset_pictures)
    questions = multi_clothing_generator.generate_questions()

    # 保存题目
    # with open("multi_image_face_feature_questions.json", "w") as f:
    #     json.dump(questions, f, indent=4)
    # print(f"Generated {len(questions)} questions and saved to multi_image_face_feature_questions.json")
