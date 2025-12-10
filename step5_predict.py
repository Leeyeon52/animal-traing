# step5_predict.py
import os
import random
from glob import glob
from ultralytics import YOLO
import config

def test_model():
    print(f"\n🚀 [5단계] AI 성능 테스트 (눈으로 확인하기)")

    # 1. 학습된 모델 파일 경로 (step4에서 만든 것)
    # runs/detect/wild_animal_model/weights/best.pt 에 저장되어 있음
    model_path = os.path.join("runs", "detect", "wild_animal_model", "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        print("   4단계(학습)가 정상적으로 완료되었는지 확인해주세요.")
        return

    print(f"🤖 모델 불러오는 중: {model_path}")
    model = YOLO(model_path)

    # 2. 시험용(Val) 이미지 폴더에서 랜덤으로 하나 뽑기
    val_image_dir = os.path.join(config.BASE_PATH, "final_dataset", "images", "val")
    
    # jpg, png 등 이미지 찾기
    images = glob(os.path.join(val_image_dir, "*.jpg")) + \
             glob(os.path.join(val_image_dir, "*.jpeg")) + \
             glob(os.path.join(val_image_dir, "*.png"))

    if not images:
        print("❌ 테스트할 이미지가 없습니다.")
        return

    # 랜덤 선택
    test_image = random.choice(images)
    print(f"📸 테스트 이미지 선택: {os.path.basename(test_image)}")

    # 3. 예측 실행 (Predict)
    # save=True: 결과를 사진으로 저장
    # conf=0.5: 확신이 50% 이상일 때만 표시
    results = model.predict(source=test_image, save=True, conf=0.5)

    # 4. 결과 위치 안내
    print("\n🎉 [테스트 완료] 결과 이미지가 저장되었습니다!")
    # ultralytics는 보통 'runs/detect/predict' 폴더에 저장합니다.
    # 여러 번 실행하면 predict2, predict3... 식으로 늘어납니다.
    print(f"   📂 확인 경로: runs/detect/ (가장 최신 폴더를 열어보세요)")
    
    # 윈도우라면 폴더를 바로 열어주기 (선택 사항)
    try:
        os.startfile(os.path.join("runs", "detect"))
    except:
        pass

if __name__ == "__main__":
    test_model()