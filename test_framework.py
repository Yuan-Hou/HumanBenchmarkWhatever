import json
import os
from copy import deepcopy
import pickle
from typing import Dict, List, Set, Tuple
import numpy as np
from functools import cache

from utils import ask_question, key_points_to_bounding_box, bounding_box_iou
from thefuzz import fuzz

DATASET_PATH = os.getenv("DATASET_PATH", "./final_labeling")

FACE_ATTR_NAMES = ['5 oClock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes', 'Bald', 'Bangs', 'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair', 'Bushy Eyebrows', 'Chubby', 'Double Chin', 'Eyeglasses', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones', 'Male', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard', 'Oval Face', 'Pale Skin', 'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks', 'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair', 'Wearing Earrings', 'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace', 'Wearing Necktie', 'Young']
FACE_ATTR_ADMIT_THRESHOLD = np.array([0.80, 0.70, 0.80, 0.50, 0.20, 0.90,                                   0.70,       0.95,       0.50,        0.50,         0.70,     0.70,         0.80,             0.95,      0.80,         0.92,         0.50,      0.50,        0.75,           0.80,              0.98,   0.995,                0.30,        0.75,         0.98,       0.70,        0.40,        0.40,           0.60,                0.14,         0.60,        0.80,      0.70,            0.70,        0.70,                0.75,         0.60,               0.50,               0.70,              0.98])
FACE_ATTR_DENY_THRESHOLD =  np.array([0.10, 0.05, 0.02, 0.05, 0.01, 0.05,                                   0.05,       0.05,       0.005,       0.01,         0.05,     0.01,         0.30,             0.40,      0.10,         0.005,        0.02,      0.005,       0.003,          0.01,              0.002,  0.02,                 0.005,       0.08,         0.40,       0.08,        0.004,       0.02,           0.01,                0.001,        0.02,        0.04,      0.05,            0.05,        0.02,                0.001,        0.01,               0.01,               0.001,             0.50])
POSITION_SIMPLIFIER = {
    'headscarf': "head",
    'shoulder': "body", 
    'ears': "head",
    'thighs': "thigh", 
    'hands': "hand",
    'right eye': "face",
    'right ear': "head",
    'right half of the face': "face", 
    'mirror': "hand", 
    'nose': "face", 
    'forehead': "head", 
    'eyes': "face", 
    'wrists': "hand", 
    'arm': "hand", 
    'tongue': "face", 
    'lip': "face", 
    'eyebrows': "face", 
    'himself': "body", 
    'back':"body", 
    'left arm': "hand", 
    'reins': "hand",
    'hair': "head", 
    'lap': "thigh", 
    'mouth': "face", 
    'left shoulder': "body", 
    'pen': "hand", 
    'mask': "face", 
    'left chest': "body"
}
POSITION_INCLUDE_MAP = {
    "hand": ["left hand", "right hand", "both hands"],
    "body": ["legs", "thigh"],
    "head": ["face", "neck"],
    "face": ["head"],
    "neck": ["head"],
    "legs": ["body"],
    "thigh": ["body"]
}
POSITION_EXCLUDE_MAP = {
    "hand": ["left hand", "right hand", "both hands", "body", "thigh"],
    "body": ["legs", "thigh"],
    "head": ["face", "neck"],
    "left hand": ["hand", "both hands", "body", "thigh"],
    "right hand": ["hand", "both hands", "body", "thigh"],
    "face": ["head"],
    "neck": ["head"]
}


_full_data = None

class HoiObject:
    def __init__(self, data):
        self.raw_data = data
        self.box = data.get("box", None)
    def get_name(self):
        return self.raw_data.get("name", "")

class Hoi:
    def __init__(self, data, obj: HoiObject):
        self.raw_data = data
        self.obj: HoiObject = obj
    def get_actions(self):
        return set([i[1] for i in self.raw_data.get("action", [])])
    def get_positions(self):
        org_pose = set([i[0] for i in self.raw_data.get("action", [])])
        return set([POSITION_SIMPLIFIER.get(p, p) for p in org_pose])
    def get_object_box(self):
        return self.obj.box
    def get_object_names(self):
        return self.obj.raw_data.get("possible_names", [])
    def get_object_name(self):
        return self.obj.raw_data.get("name", "")
    def get_negative_actions(self):
        return self.raw_data.get("negative_action", [])
    def get_position_action_pairs(self):
        return set((POSITION_SIMPLIFIER.get(i[0], i[0]), i[1]) for i in self.raw_data.get("action", []))

class Person:
    def __init__(self, data, detect_results):
        self.raw_data = data
        self.hois: List[Hoi] = []
        if data.get("without_face") is not True and data.get("face_box") is not None:
            self.face_box:List[float] = detect_results["face_boxes"][data.get("face_box")]
        else:
            self.face_box = None

        if data.get("body_box") is not None:
            self.body_box:List[float] = detect_results["body_boxes"][data.get("body_box")]
        else:
            self.body_box = None

        if data.get("skeleton") is not None:
            self.skeleton:List[List[float]] = detect_results["skeletons"][data.get("skeleton")]
        else:
            self.skeleton = None

    def init_hoi_objects(self, objs: list[HoiObject]):
        for hoi in self.raw_data.get("hoi", []):
            if "no interaction" in [i[1] for i in hoi["relationship"]["action"]]:
                continue

            if hoi.get("deleted") is not True and objs[hoi.get("object")] is not None:
                self.hois.append(Hoi(hoi["relationship"], objs[hoi.get("object")]))

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

    def get_clothing_list(self, only_confident = False):
        clothings = self.raw_data.get("qwen_detailing", {}).get("clothing", [])
        if isinstance(clothings, list):
            clothings = clothings
        elif isinstance(clothings, dict):
            if only_confident and clothings["vague"]:
                clothings = []
            else:
                clothings = clothings["clothing"]
        if only_confident:
            clothings = [c for c in clothings if c.get("belonging_confident", True) and c.get("existence_confident", True)]
        return clothings
    
    def full_feature_set(self) -> List[Tuple[Dict]]:
        """获取完整特征集合"""
        feature_set = []
        # 面部特征
        if self.face_box is not None and self.raw_data.get("facex_detailing") and self.detailing_property("face_seen", False):
            for attr_name, attr_value, accept_thresh, deny_thresh in zip(FACE_ATTR_NAMES, self.get_face_attr_vec(), FACE_ATTR_ADMIT_THRESHOLD, FACE_ATTR_DENY_THRESHOLD):
                # 只保留纯面部特征，防打架
                if attr_name not in ['5 oClock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes', 'Bald', 'Bangs', 'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair', 'Bushy Eyebrows', 'Chubby', 'Double Chin', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard', 'Oval Face', 'Pale Skin', 'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks', 'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair']:
                    continue
                if attr_value >= accept_thresh:
                    feature_set.append( {"attr_type":"facial", "attr_name": attr_name, "attr_value": True} )
                elif attr_value < deny_thresh:
                    feature_set.append( {"attr_type":"facial", "attr_name": attr_name, "attr_value": False} )
                else:
                    feature_set.append( {"attr_type":"facial", "attr_name": attr_name, "attr_value": None} )
            # 面部landmark
            if self.skeleton is not None:
                facex_point_set = np.array(self.raw_data["facex_detailing"]["landmarks"])
                wpose_point_set = np.array(self.skeleton["dw_face"])
                facex_nose = key_points_to_bounding_box(facex_point_set[[27,28,29,30,31,32,33,34,35]])  # facex鼻子相关点
                wpose_nose = key_points_to_bounding_box(wpose_point_set[[27,28,29,30,31,32,33,34,35]])  # wpose鼻子相关点
                if bounding_box_iou(facex_nose, wpose_nose) > 0.5:
                    feature_set.append( {"attr_type":"bbox", "attr_name": "nose", "attr_value": facex_nose} )
                facex_mouth = key_points_to_bounding_box(facex_point_set[[48,49,50,51,52,53,54,55,56,57,58,59]])  # facex嘴巴相关点
                wpose_mouth = key_points_to_bounding_box(wpose_point_set[[48,49,50,51,52,53,54,55,56,57,58,59]])  # wpose嘴巴相关点
                if bounding_box_iou(facex_mouth, wpose_mouth) > 0.5:
                    feature_set.append( {"attr_type":"bbox", "attr_name": "mouth", "attr_value": facex_mouth} )
                facex_leye = key_points_to_bounding_box(facex_point_set[[42,43,44,45,46,47]])  # facex左眼相关点
                wpose_leye = key_points_to_bounding_box(wpose_point_set[[42,43,44,45,46,47]])  # wpose左眼相关点
                if bounding_box_iou(facex_leye, wpose_leye) > 0.5:
                    feature_set.append( {"attr_type":"bbox", "attr_name": "left_eye", "attr_value": facex_leye} )
                facex_reye = key_points_to_bounding_box(facex_point_set[[36,37,38,39,40,41]])  # facex右眼相关点
                wpose_reye = key_points_to_bounding_box(wpose_point_set[[36,37,38,39,40,41]])  # wpose右眼相关点
                if bounding_box_iou(facex_reye, wpose_reye) > 0.5:
                    feature_set.append( {"attr_type":"bbox", "attr_name": "right_eye", "attr_value": facex_reye} )
                facex_leyebrow = key_points_to_bounding_box(facex_point_set[[22,23,24,25,26]])  # facex左眉相关点
                wpose_leyebrow = key_points_to_bounding_box(wpose_point_set[[22,23,24,25,26]])  # wpose左眉相关点
                if bounding_box_iou(facex_leyebrow, wpose_leyebrow) > 0.5:
                    feature_set.append( {"attr_type":"bbox", "attr_name": "left_eyebrow", "attr_value": facex_leyebrow} )
                facex_reyebrow = key_points_to_bounding_box(facex_point_set[[17,18,19,20,21]])  # facex右眉相关点
                wpose_reyebrow = key_points_to_bounding_box(wpose_point_set[[17,18,19,20,21]])  # wpose右眉相关点
                if bounding_box_iou(facex_reyebrow, wpose_reyebrow) > 0.5:
                    feature_set.append( {"attr_type":"bbox", "attr_name": "right_eyebrow", "attr_value": facex_reyebrow} )
            # 头部姿态
            if self.raw_data["facex_detailing"]["headpose"]["pitch"] < -15:
                feature_set.append( {"attr_type":"facial", "attr_name": "pitch", "attr_value": "down", "real_value": self.raw_data["facex_detailing"]["headpose"]["pitch"]} )
            elif self.raw_data["facex_detailing"]["headpose"]["pitch"] > 15:
                feature_set.append( {"attr_type":"facial", "attr_name": "pitch", "attr_value": "up", "real_value": self.raw_data["facex_detailing"]["headpose"]["pitch"]} )
            else:
                feature_set.append( {"attr_type":"facial", "attr_name": "pitch", "attr_value": None, "real_value": self.raw_data["facex_detailing"]["headpose"]["pitch"]} )

            if self.raw_data["facex_detailing"]["headpose"]["yaw"] < -15:
                feature_set.append( {"attr_type":"facial", "attr_name": "yaw", "attr_value": "left", "real_value": self.raw_data["facex_detailing"]["headpose"]["yaw"]} )
            elif self.raw_data["facex_detailing"]["headpose"]["yaw"] > 15:
                feature_set.append( {"attr_type":"facial", "attr_name": "yaw", "attr_value": "right", "real_value": self.raw_data["facex_detailing"]["headpose"]["yaw"]} )
            else:
                feature_set.append( {"attr_type":"facial", "attr_name": "yaw", "attr_value": None, "real_value": self.raw_data["facex_detailing"]["headpose"]["yaw"]} )
            # 面部全框
            feature_set.append( {"attr_type":"bbox", "attr_name": "face", "attr_value": self.face_box} )

        # qwen 捕获特征
        if self.raw_data.get("qwen_detailing"):
            for key in ["age", "gender", "emotion", "race"]:
                feature_set.append( {"attr_type":"overall", "attr_name": key, "attr_value": None if self.raw_data["qwen_detailing"][key] in ["unknown", "complex"] else self.raw_data["qwen_detailing"][key]} )
            if self.raw_data["qwen_detailing"].get("text") != "no_text":
                feature_set.append( {"attr_type":"overall", "attr_name": "text", "attr_value": self.raw_data["qwen_detailing"]["text"]} )

        # 衣着特征
        for clothing in self.get_clothing_list(only_confident=True):
            feature_set.append( {"attr_type":"clothing", "attr_name": "clothing", "attr_value": {"name": clothing["name"], "color": clothing["color"], "type": clothing["type"]}} )

        # 人体全框
        if self.body_box is not None:
            feature_set.append( {"attr_type":"bbox", "attr_name": "body", "attr_value": self.body_box} )

        # 人-物交互特征
        for hoi in self.hois:
            feature_set.append( {"attr_type":"hoi", "attr_name": "hoi", "attr_value": {"relation": hoi.get_position_action_pairs(), "object": hoi.get_object_name(), "bbox": hoi.get_object_box()}} )
        return feature_set
    
    def hand_cant_swap(self):
        """是否存在一件物品，左手右手都拿有"""
        left_hand_items = set()
        right_hand_items = set()
        for hoi in self.hois:
            for pos, action in hoi.get_position_action_pairs():
                if pos in ["left hand"]:
                    left_hand_items.add(hoi.get_object_name())
                if pos in ["right hand"]:
                    right_hand_items.add(hoi.get_object_name())
        return len(left_hand_items & right_hand_items) > 0

class Picture:
    def __init__(self, data):
        self.raw_data = data
        self.persons:List[Person] = [Person(p, data["detect_results"]) for p in data.get("persons", []) if p.get("deleted") is not True]
        self.hoi_objects: List[HoiObject] = []
        for obj in data.get("objects", []):
            if obj.get("deleted") is not True:
                self.hoi_objects.append(HoiObject(obj))
            else:
                self.hoi_objects.append(None)
        for person in self.persons:
            person.init_hoi_objects(self.hoi_objects)

    def image_path(self):
        return os.path.join(DATASET_PATH, self.raw_data.get("image_path").split("/")[-1])

    def full_hoi(self):
        result = []
        for person in self.persons:
            result.extend(person.hois)
        return result

    def object_names(self):
        result = []
        for obj in self.hoi_objects:
            if obj is not None:
                result.append(obj.get_name())
        return result

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

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError
# ================== 题型生成器基类 ==================

class QuestionGenerator:
    """题目生成器基类"""
    
    def __init__(self, dataset_pictures):
        self.dataset_pictures: List[Picture] = dataset_pictures
        self.picture_occurrence: Dict[Picture, int] = {}
    
    def filter_pictures(self):
        """过滤图片，子类需要重写此方法"""
        raise NotImplementedError("Subclasses must implement filter_pictures method")
    
    def generate_questions(self):
        """生成题目，子类需要重写此方法"""
        raise NotImplementedError("Subclasses must implement generate_questions method")
    
    def save_questions(self, questions, filename):
        """保存题目到文件"""
        with open(filename, "w") as f:
            json.dump(questions, f, indent=4, default=set_default)
        print(f"Generated {len(questions)} questions and saved to {filename}")
