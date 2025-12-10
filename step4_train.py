# step4_train.py
import os
from ultralytics import YOLO
import config

def train_yolo():
    print(f"\n🚀 [4단계] YOLOv11 모델 학습 시작!")
    
    # 1. data.yaml 파일 경로 찾기
    yaml_path = os.path.join(config.BASE_PATH, "final_dataset", "data.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"❌ 설정 파일이 없습니다: {yaml_path}")
        print("   3단계(데이터셋 분할)를 먼저 실행해주세요.")
        return

    print(f"📂 데이터 설정 파일: {yaml_path}")

    # 2. 모델 불러오기
    # yolo11n.pt : Nano 모델 (가장 빠르고 가벼움 -> 모바일 앱용으로 추천)
    # yolo11s.pt : Small 모델 (조금 더 정확하지만 느림)
    print("🤖 모델 초기화 중 (YOLO11 Nano)...")
    model = YOLO('yolo11n.pt') 

    # 3. 학습 시작 (Training)
    # epochs: 학습 반복 횟수 (처음엔 10으로 테스트, 실전은 50~100 추천)
    # imgsz: 이미지 크기 (640이 표준)
    # batch: 한 번에 공부할 양 (컴퓨터가 버벅이면 줄이세요: 16 -> 8 -> 4)
    print("🔥 학습을 시작합니다! (시간이 걸릴 수 있습니다)")
    
    results = model.train(
        data=yaml_path,   # 데이터 설정 파일 경로
        epochs=10,        # 반복 횟수 (테스트용 10)
        imgsz=640,        # 이미지 크기
        batch=16,         # 메모리 오류나면 8로 줄이세요
        name='wild_animal_model', # 결과가 저장될 폴더 이름
        exist_ok=True,    # 덮어쓰기 허용
        device='0' if is_gpu_available() else 'cpu' # GPU 자동 감지
    )

    print("\n🎉 [학습 완료] 축하합니다! 나만의 AI 모델이 완성되었습니다.")
    print(f"   💾 모델 파일 위치: runs/detect/wild_animal_model/weights/best.pt")
    print("   (이 best.pt 파일을 스마트폰 앱에 넣으면 됩니다!)")

def is_gpu_available():
    # GPU(NVIDIA)가 있는지 확인하는 간단한 함수
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            print("✅ GPU(그래픽카드)를 사용하여 빠르게 학습합니다.")
        else:
            print("⚠️ GPU를 찾을 수 없어 CPU로 학습합니다. (속도가 느릴 수 있습니다)")
        return available
    except:
        return False

if __name__ == "__main__":
    train_yolo()