# config.py
import os
import pandas as pd  # CSV 읽기용 (pip install pandas openpyxl 필요)

# =========================================================
# [사용자 설정] 경로를 환경에 맞게 수정하세요!
# =========================================================

# 1. 데이터가 있는 원본 폴더 (ZIP 파일들이 있는 곳)
BASE_PATH = "/workspace/175.야생동물 활동 영상 데이터"

# 2. 압축을 풀어서 저장할 폴더 (여기에 이미지들이 생깁니다)
DATASET_DIR = os.path.join(BASE_PATH, "dataset")

# 3. 변환된 라벨(txt)을 저장할 폴더
LABEL_OUTPUT_DIR = "./converted_labels"

# 4. 압축 해제 비율 (0.3 = 30%만 사용, 1.0 = 전체 사용)
EXTRACT_RATIO = 0.3

# =========================================================
# [핵심] 동물 이름 매핑 (AI-Hub + CSV 통합)
# =========================================================

# 1. AI-Hub 데이터셋에 들어있는 11종 (기본 탑재)
# 데이터셋의 영문명/학명을 우리말 ID로 연결합니다.
CLASS_MAP = {
    # 포유류
    "Hydropotes inermis": 0, "inermis": 0, "Elk": 0, "Water Deer": 0, # 고라니
    "Sus scrofa": 1, "scrofa": 1, "Wild Boar": 1,                     # 멧돼지
    "Ursus thibetanus": 2, "Asiatic Black Bear": 2,                   # 반달가슴곰
    "Nyctereutes procyonoides": 3, "Raccoon Dog": 3,                  # 너구리
    "Sciurus vulgaris": 4, "Red Squirrel": 4,                         # 청설모
    "Tamias sibiricus": 5, "Chipmunk": 5,                             # 다람쥐
    "Capreolus pygargus": 6, "Roe Deer": 6,                           # 노루
    "Lepus coreanus": 7, "Korean Hare": 7,                            # 멧토끼
    "Mustela sibirica": 8, "Weasel": 8,                               # 족제비
    
    # 조류
    "Ardea cinerea": 9, "Grey Heron": 9,                              # 왜가리
    "Ardea alba": 10, "Great Egret": 10, "Egretta alba": 10,          # 중대백로
}

# 2. 업로드한 CSV 파일에서 추가 생물 불러오기 (선택 사항)
# 가지고 계신 '멸종위기 야생생물 등급별 종 목록.csv' 경로를 입력하면 자동으로 추가됩니다.
CSV_FILE_PATH = r"붙임.멸종위기 야생생물 등급별 종 목록.xlsx - 멸종위기 야생생물.csv"

def load_csv_species():
    """CSV 파일이 있으면 읽어서 CLASS_MAP에 추가하는 함수"""
    if os.path.exists(CSV_FILE_PATH):
        try:
            # CSV 읽기 (인코딩: euc-kr 혹은 utf-8)
            try:
                df = pd.read_csv(CSV_FILE_PATH, encoding='cp949')
            except:
                df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8')
            
            # 현재 ID의 다음 번호부터 시작
            current_id = max(CLASS_MAP.values()) + 1
            
            count = 0
            # '학명' 컬럼이 있다고 가정하고 추가
            if '학명' in df.columns:
                for name in df['학명'].dropna():
                    clean_name = name.strip()
                    if clean_name not in CLASS_MAP:
                        CLASS_MAP[clean_name] = current_id
                        current_id += 1
                        count += 1
            print(f"✅ CSV 파일에서 {count}개의 멸종위기종을 리스트에 추가했습니다.")
            
        except Exception as e:
            print(f"⚠️ CSV 파일 읽기 실패 (건너뜀): {e}")
    else:
        print("ℹ️ 추가 CSV 파일이 없어 기본 11종만 처리합니다.")

# 모듈 로드 시 실행
load_csv_species()