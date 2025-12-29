from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np

# Load the trained CNN model
model = load_model('mnist.h5')

# Function to predict the digit
def predict_digit(img):
    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert the image to grayscale
    img = img.convert('L')
    img = np.array(img)
    
    # Check the shape of the image (should be (28, 28))
    print("Image shape before reshape:", img.shape)
    
    # Reshape to support the model input and normalize
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    
    # Check the shape after reshaping
    print("Image shape after reshape:", img.shape)
    
    # Predict the digit
    res = model.predict(img)[0]
    print("Prediction raw output:", res)  # Debugging step
    
    # Return the predicted digit and confidence
    return np.argmax(res), max(res)

# Create the Tkinter application class
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # Create canvas and buttons for the GUI
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Arrange the components using grid
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # Bind the mouse event for drawing
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    # Function to clear the canvas
    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="Draw..")  # Reset label when clearing

    # Function to classify the handwriting and display the result
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # Get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # Get the coordinates of the canvas
        a, b, c, d = rect
        rect = (a+4, b+4, c-4, d-4)  # Adjust the canvas boundaries
        im = ImageGrab.grab(rect)  # Capture the drawing on the canvas

        # Predict the digit
        digit, acc = predict_digit(im)
        self.label.configure(text=f'{digit}, {int(acc * 100)}%')  # Display the predicted digit and accuracy

    # Function to draw lines on the canvas
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8  # Radius of the circle to be drawn
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill='black')

# Run the Tkinter application
app = App()
mainloop()
