# step1_unzip.py
import os
import zipfile
import random
from glob import glob
from tqdm import tqdm
import config  # 설정 불러오기

def unzip_ratio():
    print(f"\n🚀 [1단계] 데이터 압축 해제 시작 (비율: {config.EXTRACT_RATIO*100}%)")
    print(f"📂 원본 경로: {config.BASE_PATH}")
    print(f"📂 저장 경로: {config.DATASET_DIR}")

    # 1. ZIP 파일 탐색
    zip_files = glob(os.path.join(config.BASE_PATH, "**", "*.zip"), recursive=True)
    if not zip_files:
        print("❌ ZIP 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f"   -> 총 {len(zip_files)}개의 ZIP 파일 발견.")

    # 2. 파일별 압축 해제
    for zip_path in tqdm(zip_files, desc="진행 중"):
        try:
            # 폴더명 정리 (파일명으로 폴더 생성)
            folder_name = os.path.splitext(os.path.basename(zip_path))[0]
            target_dir = os.path.join(config.DATASET_DIR, folder_name)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                
                # 이미지/라벨 짝(Pair) 맞추기 위해 이름(stem)만 추출
                file_stems = list(set(os.path.splitext(f)[0] for f in all_files))
                
                # 설정된 비율만큼 랜덤 추출
                target_count = int(len(file_stems) * config.EXTRACT_RATIO)
                if target_count < 1: target_count = 1 # 최소 1개는 해제
                
                selected_stems = set(random.sample(file_stems, target_count))

                # 선택된 이름이 포함된 파일만 리스트업
                files_to_extract = [
                    f for f in all_files 
                    if os.path.splitext(f)[0] in selected_stems
                ]
                
                if files_to_extract:
                    zip_ref.extractall(path=target_dir, members=files_to_extract)
                
        except zipfile.BadZipFile:
            print(f"\n⚠️ 손상된 ZIP 파일 건너뜀: {zip_path}")
        except Exception as e:
            print(f"\n⚠️ 오류 발생 ({zip_path}): {e}")

    print("\n✅ [1단계 완료] 압축 해제가 끝났습니다.")

if __name__ == "__main__":
    unzip_ratio()