import json
import os
from glob import glob
from tqdm import tqdm

# ==========================================
# [1. 설정] 다운로드 받은 폴더 경로를 넣어주세요
# ==========================================
base_path = r"D:\175.야생동물 활동 영상 데이터"  # 데이터가 있는 최상위 폴더
output_folder = "./converted_labels" # 변환된 파일이 저장될 폴더

# [2. 동물 이름 매칭] 데이터에 있는 '학명'을 숫자 ID로 바꿉니다.
# 텍스트에 나온 'category_String'을 기준으로 매핑합니다.
class_map = {
    # 학명(데이터 내용) : ID(우리가 정한 번호) : 한국어 설명
    "Hydropotes inermis": 0, # 고라니 (inermis로만 되어있을 수도 있음)
    "inermis": 0,            # (혹시 몰라 추가)
    "Sus scrofa": 1,         # 멧돼지
    "scrofa": 1,
    "Ursus thibetanus": 2,   # 반달가슴곰
    "Nyctereutes procyonoides": 3, # 너구리
    # 필요하면 청설모, 다람쥐 등 추가
}

os.makedirs(output_folder, exist_ok=True)

# 모든 JSON 파일 찾기
json_files = glob(os.path.join(base_path, "**/*.json"), recursive=True)
print(f"총 {len(json_files)}개의 파일을 찾았습니다.")

for json_file in tqdm(json_files):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지 크기 정보 찾기 (정규화를 위해 필수)
        # 구조: images 리스트 안에 정보가 있음
        img_info = data['images'][0]
        img_width = img_info['width']
        img_height = img_info['height']
        file_name = img_info['file_name']

        # 저장할 txt 파일 이름 만들기 (이미지 파일명과 동일하게)
        txt_filename = os.path.splitext(file_name)[0] + ".txt"
        
        yolo_lines = []

        # 어노테이션(정답지) 순회
        for anno in data['annotations']:
            # 동물 이름 확인 (speciesString 또는 category_String 필드 확인 필요)
            # 제공해주신 텍스트에는 'speciesString'과 'category_String'이 둘 다 보임
            # 우선순위로 확인
            species_name = anno.get('speciesString') or anno.get('category_String')
            
            # 우리가 찾는 동물이 아니면 건너뛰기
            if species_name not in class_map:
                continue

            class_id = class_map[species_name]

            # 좌표 변환: [[x1, y1], [x2, y2]] -> YOLO(centerX, centerY, w, h)
            # AI-Hub 명세서에 bbox가 이중 리스트[[x1,y1],[x2,y2]]라고 되어 있음
            box = anno['bbox'] 
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[1][0], box[1][1]

            # 중심점과 너비, 높이 계산
            w_abs = x2 - x1
            h_abs = y2 - y1
            x_center_abs = x1 + (w_abs / 2)
            y_center_abs = y1 + (h_abs / 2)

            # 0~1 사이로 정규화 (YOLO 필수)
            x_center = x_center_abs / img_width
            y_center = y_center_abs / img_height
            w = w_abs / img_width
            h = h_abs / img_height

            # 한 줄 추가
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # 변환된 내용이 있으면 저장
        if yolo_lines:
            with open(os.path.join(output_folder, txt_filename), 'w') as f:
                f.write('\n'.join(yolo_lines))

    except Exception as e:
        # 에러가 나면 어떤 파일인지 출력
        print(f"\n[Error] {json_file} 처리 중 오류: {e}")

print("변환 완료! 이제 학습을 시작할 수 있습니다.")