import cv2
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model

# ==============================
# DEVICE SETUP
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# TEXT EMOTION MODEL (30)
# ==============================

text_model = None
tokenizer = None

text_emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral", "calm", "frustration"
]

def load_text_model():
    global text_model, tokenizer

    if text_model is not None:
        return

    print("Loading Text Emotion Model...")

    model_name = "monologg/bert-base-cased-goemotions-original"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    text_model.to(device)
    text_model.eval()

    print("Text Model Loaded Successfully!")

def analyze_text():
    load_text_model()

    sentence = input("Enter text: ")

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = text_model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0]
    top = torch.topk(probs, 3)

    print("\nTop 3 Emotions:")
    for idx, score in zip(top.indices, top.values):
        print(f"{text_emotions[idx]} → {score.item():.2f}")

# ==============================
# FACE EMOTION MODEL (FER2013)
# ==============================

print("Loading Face Emotion Model...")
face_model = load_model("emotion_model.hdf5")

face_emotions = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def analyze_face():
    cap = cv2.VideoCapture(0)
    print("Press 'x' to exit webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))

            preds = face_model.predict(roi, verbose=0)
            emotion = face_emotions[np.argmax(preds)]

            cv2.rectangle(frame, (x, y),
                          (x+w, y+h),
                          (0, 255, 0), 2)

            cv2.putText(frame,
                        emotion,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2)

        cv2.imshow("Multimodal Emotion AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==============================
# MAIN MENU
# ==============================

def main():
    while True:
        print("\n===== MULTIMODAL EMOTION AI =====")
        print("1 → Text Emotion (30 classes)")
        print("2 → Face Emotion (7 classes)")
        print("3 → Exit")

        try:
            choice = int(input("Enter choice: "))

            if choice == 1:
                analyze_text()
            elif choice == 2:
                analyze_face()
            elif choice == 3:
                print("Exiting...")
                break
            else:
                print("Invalid option.")

        except ValueError:
            print("Enter valid number.")

if __name__ == "__main__":
    main()
