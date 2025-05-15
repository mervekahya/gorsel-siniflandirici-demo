import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Model sınıfını yeniden tanımlamalıyız
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)  # 10 sınıf
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Modeli yükle
model_path = "model/animal_classifier.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Sınıf isimleri (alphabetik sıraya göre ImageFolder'dan gelir)
class_names = ['dog', 'horse', 'elephant', 'cow', 'butterfly', 'chichen', 'cat', 'cow', 'sheep', 'spider','squirrel']

# Görüntü ön işleme
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

st.title("🐾 Animals-10 Görüntü Sınıflandırıcı")
st.info(
    "🔍 Bu yapay zeka modeli yalnızca aşağıdaki 10 hayvanı tanıyabilir:\n\n"
    "- 🦋 **Butterfly**\n"
    "- 🐱 **Cat**\n"
    "- 🐔 **Chicken**\n"
    "- 🐄 **Cow**\n"
    "- 🐶 **Dog**\n"
    "- 🐘 **Elephant**\n"
    "- 🐴 **Horse**\n"
    "- 🐑 **Sheep**\n"
    "- 🕷️ **Spider**\n"
    "- 🐿️ **Squirrel**"
)




uploaded_file = st.file_uploader("Görsel Yükle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # Tahmin butonu
    if st.button("Tahmin Et"):
        with st.spinner("Tahmin ediliyor..."):
            img = transform(image).unsqueeze(0).to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            class_name = class_names[predicted.item()]
            st.success(f"📢 Bu bir **{class_name.upper()}**!")
