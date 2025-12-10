import os
import json
import zipfile
import random
from glob import glob
from tqdm import tqdm

# =========================================================
# [설정] 경로 및 추출 비율 설정
# =========================================================
base_path = r"D:\175.야생동물 활동 영상 데이터"
output_folder = "./converted_labels"
EXTRACT_RATIO = 0.3  # 30%만 해제 (0.1로 하면 10%만 해제)

# [설정] 동물 ID 매핑
class_map = {
    "Hydropotes inermis": 0, # 고라니
    "inermis": 0,
    "Sus scrofa": 1,         # 멧돼지
    "scrofa": 1,
    "Ursus thibetanus": 2,   # 반달가슴곰
    "Nyctereutes procyonoides": 3, # 너구리
}

# =========================================================
# [1단계] 30% 랜덤 압축 해제 함수 (짝 맞춤 기능 포함)
# =========================================================
def unzip_ratio_files(root_path, ratio):
    print(f"\n🚀 [1단계] 각 ZIP 파일에서 {ratio*100}%만 랜덤 추출합니다...")

    zip_files = glob(os.path.join(root_path, "**", "*.zip"), recursive=True)
    
    if not zip_files:
        print("   -> 압축 파일(.zip)이 없습니다.")
        return

    print(f"   -> 총 {len(zip_files)}개의 ZIP 파일을 발견했습니다.")

    for zip_path in tqdm(zip_files, desc="압축 해제 중"):
        try:
            extract_path = os.path.dirname(zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 1. 압축 파일 내의 모든 파일 목록 가져오기
                all_files = zip_ref.namelist()
                
                # 2. 파일명(확장자 제외)끼리 그룹 묶기 (jpg와 json을 한 쌍으로 묶기 위함)
                # 예: 'image_01.jpg', 'image_01.json' -> 'image_01'
                file_stems = list(set(os.path.splitext(f)[0] for f in all_files))
                
                # 3. 그룹 중에서 30%만 랜덤 선택
                target_count = int(len(file_stems) * ratio)
                if target_count == 0: target_count = 1 # 최소 1개는 풀기
                
                selected_stems = random.sample(file_stems, target_count)
                selected_stems_set = set(selected_stems) # 검색 속도를 위해 집합으로 변환

                # 4. 선택된 이름이 포함된 파일들만 리스트업
                # (파일명이 selected_stems_set에 포함된 경우에만 추출 목록에 추가)
                files_to_extract = [
                    f for f in all_files 
                    if os.path.splitext(f)[0] in selected_stems_set
                ]
                
                # 5. 선택된 파일만 압축 해제
                # (이미 파일이 있으면 건너뛰는 로직은 복잡해지므로 덮어쓰기 진행)
                if files_to_extract:
                    zip_ref.extractall(path=extract_path, members=files_to_extract)
                
        except Exception as e:
            print(f"\n[Error] {zip_path} 처리 실패: {e}")

    print("✅ 부분 압축 해제 완료!\n")

# =========================================================
# [2단계] JSON -> YOLO 변환 함수 (이전과 동일)
# =========================================================
def convert_json_to_yolo(root_path, save_path):
    print(f"🚀 [2단계] 추출된 데이터 변환 시작 (JSON -> YOLO)...")
    
    os.makedirs(save_path, exist_ok=True)
    json_files = glob(os.path.join(root_path, "**", "*.json"), recursive=True)
    
    if not json_files:
        print("❌ 변환할 JSON 파일이 없습니다.")
        return

    print(f"   -> 변환 대상 파일: {len(json_files)}개")
    converted_count = 0
    
    for json_file in tqdm(json_files, desc="라벨 변환 중"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'images' not in data or not data['images']: continue

            img_info = data['images'][0]
            img_width = img_info.get('width')
            img_height = img_info.get('height')
            file_name = img_info.get('file_name')
            
            if not img_width or not file_name: continue

            txt_filename = os.path.splitext(file_name)[0] + ".txt"
            yolo_lines = []

            if 'annotations' in data:
                for anno in data['annotations']:
                    species = anno.get('speciesString') or anno.get('category_String') or anno.get('category_name')
                    
                    if species not in class_map: continue
                    class_id = class_map[species]

                    bbox = anno.get('bbox')
                    if not bbox: continue

                    if isinstance(bbox[0], list):
                        x1, y1 = bbox[0][0], bbox[0][1]
                        x2, y2 = bbox[1][0], bbox[1][1]
                        w_abs = x2 - x1
                        h_abs = y2 - y1
                        x_center_abs = x1 + (w_abs / 2)
                        y_center_abs = y1 + (h_abs / 2)
                    else: continue

                    x_center = x_center_abs / img_width
                    y_center = y_center_abs / img_height
                    w = w_abs / img_width
                    h = h_abs / img_height

                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            if yolo_lines:
                with open(os.path.join(save_path, txt_filename), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))
                converted_count += 1

        except Exception:
            pass

    print(f"✅ 작업 끝! 총 {converted_count}개의 라벨이 생성되었습니다.")
    print(f"📁 저장 위치: {os.path.abspath(save_path)}")

# =========================================================
# [실행]
# =========================================================
if __name__ == "__main__":
    unzip_ratio_files(base_path, EXTRACT_RATIO)
    convert_json_to_yolo(base_path, output_folder)