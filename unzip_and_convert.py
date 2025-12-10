import os
import json
import zipfile
from glob import glob
from tqdm import tqdm  # 진행바 표시용

# =========================================================
# [설정] 경로 설정 (역슬래시 \ 대신 슬래시 / 사용 추천)
# =========================================================
base_path = r"D:\175.야생동물 활동 영상 데이터"  
output_folder = "./converted_labels" # 변환된 라벨이 저장될 폴더

# [설정] 동물 ID 매핑 (AI-Hub 데이터 기준)
class_map = {
    "Hydropotes inermis": 0, # 고라니
    "inermis": 0,
    "Sus scrofa": 1,         # 멧돼지
    "scrofa": 1,
    "Ursus thibetanus": 2,   # 반달가슴곰
    "Nyctereutes procyonoides": 3, # 너구리
}

# =========================================================
# [1단계] 자동 압축 해제 함수
# =========================================================
def unzip_all_files(root_path):
    print(f"\n🚀 [1단계] '{root_path}' 내부의 모든 ZIP 파일을 찾습니다...")
    
    # 하위 폴더까지 모든 .zip 파일 찾기
    zip_files = glob(os.path.join(root_path, "**", "*.zip"), recursive=True)
    
    if not zip_files:
        print("   -> 압축 파일(.zip)이 없습니다. 이미 풀려있거나 경로가 틀렸을 수 있습니다.")
        return

    print(f"   -> 총 {len(zip_files)}개의 압축 파일을 발견했습니다. 해제를 시작합니다.")

    for zip_path in tqdm(zip_files, desc="압축 해제 중"):
        try:
            # 압축 파일이 있는 그 폴더에 바로 풉니다
            extract_path = os.path.dirname(zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 이미 풀린 파일이 있는지 체크하면 좋겠지만, 
                # 간단하게는 덮어쓰기 모드로 풉니다.
                zip_ref.extractall(extract_path)
                
        except Exception as e:
            print(f"\n[Error] {zip_path} 해제 실패: {e}")

    print("✅ 압축 해제 완료!\n")

# =========================================================
# [2단계] JSON -> YOLO 변환 함수
# =========================================================
def convert_json_to_yolo(root_path, save_path):
    print(f"🚀 [2단계] JSON 라벨 데이터를 YOLO 포맷으로 변환합니다...")
    
    # 저장 폴더 생성
    os.makedirs(save_path, exist_ok=True)
    
    # 압축이 풀린 JSON 파일들 찾기
    json_files = glob(os.path.join(root_path, "**", "*.json"), recursive=True)
    
    if not json_files:
        print("❌ JSON 파일을 찾을 수 없습니다. 압축 해제가 제대로 안 되었을 수 있습니다.")
        return

    print(f"   -> 변환 대상 JSON 파일: {len(json_files)}개")
    
    converted_count = 0
    
    for json_file in tqdm(json_files, desc="라벨 변환 중"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 이미지 정보가 없거나 리스트가 비어있으면 패스
            if 'images' not in data or not data['images']:
                continue

            img_info = data['images'][0]
            img_width = img_info.get('width')
            img_height = img_info.get('height')
            file_name = img_info.get('file_name')

            if not img_width or not img_height or not file_name:
                continue

            # 파일명 확장자 변경 (.jpg -> .txt)
            txt_filename = os.path.splitext(file_name)[0] + ".txt"
            yolo_lines = []

            # 어노테이션 처리
            if 'annotations' in data:
                for anno in data['annotations']:
                    # 동물 이름 확인 (키 값이 다를 수 있어 여러 개 확인)
                    species = anno.get('speciesString') or anno.get('category_String') or anno.get('category_name')
                    
                    if species not in class_map:
                        continue

                    class_id = class_map[species]

                    # bbox 처리 [[x1,y1],[x2,y2]] 또는 [x,y,w,h] 등 확인
                    bbox = anno.get('bbox')
                    if not bbox:
                        continue

                    # AI-Hub 야생동물 데이터 포맷 [[x1, y1], [x2, y2]] 처리
                    if isinstance(bbox[0], list):
                        x1, y1 = bbox[0][0], bbox[0][1]
                        x2, y2 = bbox[1][0], bbox[1][1]
                        w_abs = x2 - x1
                        h_abs = y2 - y1
                        x_center_abs = x1 + (w_abs / 2)
                        y_center_abs = y1 + (h_abs / 2)
                    else:
                        # 혹시 다른 포맷일 경우 (x, y, w, h) 등.. 패스
                        continue

                    # 정규화
                    x_center = x_center_abs / img_width
                    y_center = y_center_abs / img_height
                    w = w_abs / img_width
                    h = h_abs / img_height

                    # 범위 체크 (0~1 사이)
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))

                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # 변환된 내용 저장
            if yolo_lines:
                with open(os.path.join(save_path, txt_filename), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))
                converted_count += 1

        except Exception as e:
            # 너무 많은 에러 로그 방지를 위해 생략하거나 필요시 출력
            pass

    print(f"✅ 모든 작업 완료! 총 {converted_count}개의 라벨 파일이 생성되었습니다.")
    print(f"📁 저장 위치: {os.path.abspath(save_path)}")

# =========================================================
# [메인 실행부]
# =========================================================
if __name__ == "__main__":
    # 1. 압축 해제 실행
    unzip_all_files(base_path)
    
    # 2. 라벨 변환 실행
    convert_json_to_yolo(base_path, output_folder)