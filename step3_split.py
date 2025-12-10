# step3_split.py
import os
import shutil
import random
import yaml  # pip install pyyaml 필요 (없으면 자동 생성 텍스트 출력)
from glob import glob
from tqdm import tqdm
import config  # 설정 불러오기

def split_dataset():
    print(f"\n🚀 [3단계] 데이터셋 분할 및 정리 (Train 8 : Val 2)")
    
    # 최종 데이터셋이 저장될 폴더
    FINAL_DIR = os.path.join(config.BASE_PATH, "final_dataset")
    
    # YOLO 표준 폴더 구조 생성
    subdirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(FINAL_DIR, subdir), exist_ok=True)

    print(f"📂 저장 위치: {FINAL_DIR}")

    # 1. 이미지 인덱싱 (속도 향상을 위해 미리 위치 파악)
    print("🔍 이미지 파일 위치를 파악하는 중...")
    image_paths = {} # { '파일명_stem': '전체경로' }
    
    # 지원할 이미지 확장자
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    for ext in exts:
        for path in glob(os.path.join(config.DATASET_DIR, "**", ext), recursive=True):
            stem = os.path.splitext(os.path.basename(path))[0]
            image_paths[stem] = path

    # 2. 라벨 파일과 매칭
    print("🔍 라벨과 이미지 짝 맞추기...")
    label_files = glob(os.path.join(config.LABEL_OUTPUT_DIR, "*.txt"))
    
    paired_data = [] # (이미지경로, 라벨경로) 튜플 리스트
    
    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        
        # 짝이 되는 이미지가 있는지 확인
        if stem in image_paths:
            img_path = image_paths[stem]
            paired_data.append((img_path, label_path))
    
    print(f"   -> 총 {len(image_paths)}개 이미지 중 {len(paired_data)}쌍의 데이터(이미지+라벨) 확보!")

    if len(paired_data) == 0:
        print("❌ 매칭된 데이터가 없습니다. 파일명을 확인해주세요.")
        return

    # 3. 랜덤 셔플 및 분할 (8:2)
    random.shuffle(paired_data)
    
    split_idx = int(len(paired_data) * 0.8)
    train_set = paired_data[:split_idx]
    val_set = paired_data[split_idx:]
    
    print(f"   -> 학습용(Train): {len(train_set)}장, 검증용(Val): {len(val_set)}장")

    # 4. 파일 복사 함수
    def copy_files(dataset, split_name):
        for img_src, label_src in tqdm(dataset, desc=f"{split_name} 복사 중"):
            # 파일명 추출
            filename = os.path.basename(img_src)
            label_name = os.path.basename(label_src)
            
            # 목적지 경로
            img_dst = os.path.join(FINAL_DIR, "images", split_name, filename)
            label_dst = os.path.join(FINAL_DIR, "labels", split_name, label_name)
            
            # 복사 (공간 절약을 위해 이동하려면 shutil.move 사용)
            shutil.copy2(img_src, img_dst)
            shutil.copy2(label_src, label_dst)

    # 실제 복사 수행
    copy_files(train_set, "train")
    copy_files(val_set, "val")

    # 5. data.yaml 파일 자동 생성
    print("\n📝 YOLO 학습 설정 파일(data.yaml) 생성 중...")
    
    # config.CLASS_MAP을 뒤집어서 {0: '고라니', 1: '멧돼지'} 형태로 만듦
    id_to_name = {v: k for k, v in config.CLASS_MAP.items() if isinstance(v, int)}
    # 중복 제거 및 정렬 (하나의 ID에 여러 이름이 있을 경우 하나만 선택)
    names_list = []
    # 0번부터 최대 ID까지 순서대로 이름 찾기
    max_id = max(config.CLASS_MAP.values())
    for i in range(max_id + 1):
        # 해당 ID를 가진 키 중 첫 번째(주로 영문명)를 찾음
        found_name = "Unknown"
        for k, v in config.CLASS_MAP.items():
            if v == i:
                found_name = k
                break
        names_list.append(found_name)

    yaml_content = {
        'path': os.path.abspath(FINAL_DIR), # 절대 경로
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(names_list)}
    }
    
    yaml_path = os.path.join(FINAL_DIR, "data.yaml")
    
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
        print(f"✅ data.yaml 생성 완료: {yaml_path}")
    except Exception as e:
        print(f"⚠️ YAML 생성 실패 (직접 만드세요): {e}")

    print("\n🎉 [3단계 완료] 모든 준비가 끝났습니다!")
    print(f"   학습 시작 시 경로: {yaml_path}")

if __name__ == "__main__":
    split_dataset()