from dotenv import load_dotenv
from test_framework import get_full_data, Picture
from multi_face_feature_generator import MultiFaceFeatureQuestionGenerator
from multi_clothing_feature_generator import MultiPersonClothingFeatureQuestionGenerator

load_dotenv()

if __name__ == "__main__":
    full_data = get_full_data()
    dataset_pictures = [Picture(p) for p in full_data]
    print(f"Loaded {len(full_data)} records from dataset.")

    # 生成多图人体服装特征题目
    multi_clothing_generator = MultiPersonClothingFeatureQuestionGenerator(dataset_pictures)
    clothing_questions = multi_clothing_generator.generate_questions()
    
    # if clothing_questions:
    #     multi_clothing_generator.save_questions(clothing_questions, "multi_clothing_feature_questions.json")

    # 生成多图人脸特征题目（目前注释掉，因为计算量较大）
    # multi_face_generator = MultiFaceFeatureQuestionGenerator(dataset_pictures)
    # face_questions = multi_face_generator.generate_questions()
    # if face_questions:
    #     multi_face_generator.save_questions(face_questions, "multi_face_feature_questions.json")
