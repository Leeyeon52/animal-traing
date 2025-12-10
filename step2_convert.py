# step2_convert.py
import os
import json
from glob import glob
from tqdm import tqdm
import config  # 설정 불러오기

def convert_labels():
    print(f"\n🚀 [2단계] JSON -> YOLO 포맷 변환 시작")
    print(f"📂 대상 경로: {config.DATASET_DIR}")
    
    os.makedirs(config.LABEL_OUTPUT_DIR, exist_ok=True)
    
    # JSON 파일 탐색
    json_files = glob(os.path.join(config.DATASET_DIR, "**", "*.json"), recursive=True)
    if not json_files:
        print("❌ 변환할 JSON 파일이 없습니다. 1단계를 먼저 실행하세요.")
        return

    print(f"   -> 처리할 파일: {len(json_files)}개")
    converted_count = 0
    
    for json_file in tqdm(json_files, desc="변환 중"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 데이터 유효성 검사
            if 'images' not in data or not data['images']: continue
            img_info = data['images'][0]
            
            img_width = img_info.get('width')
            img_height = img_info.get('height')
            file_name = img_info.get('file_name')
            
            if not img_width or not file_name: continue

            # 결과 파일명 생성
            txt_filename = os.path.splitext(file_name)[0] + ".txt"
            yolo_lines = []

            if 'annotations' in data:
                for anno in data['annotations']:
                    # 1. 종(Species) 확인: 여러 키워드 중 하나라도 매칭되면 OK
                    species = (anno.get('speciesString') or 
                               anno.get('category_String') or 
                               anno.get('category_name'))
                    
                    # 2. 매핑된 ID 찾기
                    if species not in config.CLASS_MAP:
                        continue # 목록에 없는 동물은 무시
                    
                    class_id = config.CLASS_MAP[species]

                    # 3. 좌표(BBox) 처리
                    bbox = anno.get('bbox')
                    if not bbox: continue

                    # 포맷 1: [[x1, y1], [x2, y2]] (AI-Hub 야생동물 표준)
                    if isinstance(bbox[0], list):
                        x1, y1 = bbox[0][0], bbox[0][1]
                        x2, y2 = bbox[1][0], bbox[1][1]
                        w_abs = x2 - x1
                        h_abs = y2 - y1
                        x_center_abs = x1 + (w_abs / 2)
                        y_center_abs = y1 + (h_abs / 2)
                        
                    # 포맷 2: [x, y, w, h] (일반 COCO 포맷)
                    elif isinstance(bbox[0], (int, float)):
                        x_abs, y_abs, w_abs, h_abs = bbox
                        x_center_abs = x_abs + (w_abs / 2)
                        y_center_abs = y_abs + (h_abs / 2)
                    else:
                        continue # 알 수 없는 포맷

                    # 4. 정규화 (0~1 사이 값으로 변환)
                    x_center = x_center_abs / img_width
                    y_center = y_center_abs / img_height
                    w = w_abs / img_width
                    h = h_abs / img_height

                    # 5. 범위 제한 (가끔 좌표가 이미지 밖으로 나가는 경우 방지)
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))

                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # 파일 저장 (내용이 있을 때만)
            if yolo_lines:
                save_path = os.path.join(config.LABEL_OUTPUT_DIR, txt_filename)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))
                converted_count += 1

        except Exception:
            pass # 개별 파일 에러는 무시하고 계속 진행

    print(f"\n✅ [2단계 완료] 총 {converted_count}개의 라벨 파일 생성 완료!")
    print(f"📁 결과 확인: {os.path.abspath(config.LABEL_OUTPUT_DIR)}")

if __name__ == "__main__":
    convert_labels()