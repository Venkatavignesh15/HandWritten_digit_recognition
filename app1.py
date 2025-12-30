import tkinter as tk
from keras.models import load_model
from PIL import ImageGrab, Image
import numpy as np
import win32gui

# Load model
model = load_model("model/mnist_cnn.keras")

def predict_digit(img):
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognition")
        self.geometry("500x350")

        self.canvas = tk.Canvas(self, width=300, height=300, bg="white")
        self.canvas.grid(row=0, column=0, pady=10, padx=10)

        self.label = tk.Label(self, text="Draw a digit", font=("Arial", 24))
        self.label.grid(row=0, column=1)

        btn_predict = tk.Button(self, text="Predict", command=self.classify)
        btn_predict.grid(row=1, column=1, pady=5)

        btn_clear = tk.Button(self, text="Clear", command=self.clear)
        btn_clear.grid(row=1, column=0)

        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        r = 12
        self.canvas.create_oval(
            event.x - r, event.y - r,
            event.x + r, event.y + r,
            fill="black"
        )

    def clear(self):
        self.canvas.delete("all")
        self.label.config(text="Draw a digit")

    def classify(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        img = ImageGrab.grab(rect)

        digit, confidence = predict_digit(img)
        self.label.config(text=f"{digit} ({confidence*100:.2f}%)")

app = App()
app.mainloop()
