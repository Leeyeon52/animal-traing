from ultralytics import YOLO

# 1. 모델 불러오기 (같은 폴더에 있는 best.pt)
print("📂 모델을 불러옵니다...")
model = YOLO("best.pt")

# 2. TFLite로 변환 (스마트폰용)
# int8=True 옵션은 용량을 줄여주지만, 변환 에러가 날 수 있어 안전하게 기본(float32)으로 합니다.
print("⚡ TFLite로 변환을 시작합니다...")
model.export(format="tflite")

print("🎉 변환 완료! 폴더를 확인하세요.")