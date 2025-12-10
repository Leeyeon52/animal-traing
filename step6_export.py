# step6_export.py
import os
import shutil
from ultralytics import YOLO

def export_model():
    print(f"\n🚀 [6단계] 스마트폰용(TFLite) 변환 시작!")

    # 1. 학습된 모델 위치 (도커 내부 경로)
    # 아까 로그에 찍힌 save_dir 경로입니다.
    model_path = "/ultralytics/runs/detect/wild_animal_model/weights/best.pt"

    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        print("   학습(4단계)이 완전히 끝났는지 확인해주세요.")
        return

    print(f"📂 학습된 모델 발견: {model_path}")
    
    # 2. 모델 불러오기
    model = YOLO(model_path)

    # 3. TFLite로 변환 (Export)
    # format='tflite': 안드로이드/iOS용 포맷
    # int8=True: 용량을 4배 줄이고 속도를 높임 (모바일 필수 옵션)
    print("⚡ TFLite로 변환 중... (시간이 좀 걸립니다)")
    
    # 3-1. 일반 tflite 변환
    model.export(format='tflite') 
    
    # 4. 변환된 파일을 윈도우 폴더로 꺼내오기
    # 변환되면 best.pt가 있는 폴더에 best_saved_model/best_float32.tflite 등이 생김
    # 가장 쓰기 편한 float32 버전을 가져옵니다.
    
    source_tflite = "/ultralytics/runs/detect/wild_animal_model/weights/best_saved_model/best_float32.tflite"
    
    # 혹시 경로가 다를 수 있어서 확인
    if not os.path.exists(source_tflite):
        # 구버전 경로 등 예외 처리
        source_tflite = model_path.replace(".pt", ".tflite")

    destination = "/workspace/동물/wild_animal.tflite" # 우리가 원하는 최종 이름

    if os.path.exists(source_tflite):
        shutil.copy2(source_tflite, destination)
        print(f"\n🎉 [변환 성공] 파일이 윈도우 폴더로 복사되었습니다!")
        print(f"   💾 최종 파일: D:\\동물\\wild_animal.tflite")
        print("   (이제 이 파일을 플러터 앱에 넣으면 됩니다!)")
    else:
        print("⚠️ 변환은 된 것 같은데 파일을 못 찾겠습니다. 도커 경로를 확인해주세요.")

if __name__ == "__main__":
    export_model()