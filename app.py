import os
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from model import CNN  # CNN class from model.py

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get class names from raw-img folder
class_names = sorted(os.listdir("C:/Users/User/Desktop/Bulut/dataset/raw-img"))

# Model creation
num_classes = len(class_names)
model = CNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Transform for image processing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# UI design
def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    prediction = predict_image(file_path)
    result_label.config(text=f"Prediction: {prediction}")

    img = Image.open(file_path)
    img.thumbnail((250, 250))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Initialize Tkinter interface
root = tk.Tk()
root.title("Görüntü Sınıflandırıcı")
root.geometry("500x600")
root.configure(bg="#f0f0f0")

# Ana çerçeve
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(expand=True, fill="both", padx=20, pady=20)

# Başlık
title_label = tk.Label(
    main_frame,
    text="Görüntü Sınıflandırma Uygulaması",
    font=("Helvetica", 18, "bold"),
    bg="#f0f0f0",
    fg="#2c3e50"
)
title_label.pack(pady=20)

# Özelleştirilmiş buton
btn = tk.Button(
    main_frame,
    text="Resim Seç",
    command=open_image,
    font=("Helvetica", 12),
    bg="#3498db",
    fg="white",
    activebackground="#2980b9",
    activeforeground="white",
    pady=10,
    padx=20,
    relief="flat",
    cursor="hand2"
)
btn.pack(pady=20)

# Resim gösterme alanı
image_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief="solid")
image_frame.pack(pady=20)

image_label = tk.Label(image_frame, bg="#ffffff")
image_label.pack(padx=10, pady=10)

# Tahmin sonucu
result_label = tk.Label(
    main_frame,
    text="Tahmin: Henüz bir resim seçilmedi",
    font=("Helvetica", 14),
    bg="#f0f0f0",
    fg="#2c3e50",
    wraplength=400
)
result_label.pack(pady=20)

root.mainloop()
