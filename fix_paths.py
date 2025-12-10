import yaml
import os

# 도커 안에서의 파일 위치
# (폴더명이 정확한지 확인해주세요)
yaml_path = '/workspace/175.야생동물 활동 영상 데이터/final_dataset/data.yaml'

print(f"🔧 YAML 파일 수정 중: {yaml_path}")

try:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 1. 절대 경로를 도커 기준으로 변경
    data['path'] = '/workspace/175.야생동물 활동 영상 데이터/final_dataset'
    
    # 2. 상대 경로로 설정
    data['train'] = 'images/train'
    data['val'] = 'images/val'

    # 저장
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        
    print("✅ 주소 수정 완료! 이제 학습 코드를 실행하세요.")

except Exception as e:
    print(f"❌ 오류 발생: {e}")