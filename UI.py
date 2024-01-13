###############################################################################################################
# File Name : UI.py
# Detail    : Defines the user interface widget for the License plate detector
# Project   : LicensePlate Detector
# Author    : Ashwin Bardhwaj
# Date      : 12/26/2023
#
# @ Copyright @2023
###############################################################################################################

import tkinter as tk
from tkinter import filedialog
from functools import partial
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

window = tk.Tk()
window.title("License Plate Detector")
window.geometry("800x800")

d_label = tk.Label(window, text="Download an Image!")
d_label.grid(row=0, column=0)

def handle_button():
    file = filedialog.askopenfilename(filetypes = [('PNG File', '*.png')])
    if file is not None:
        print(file)
        fileName = getFileNameFromPath(file)
        runPrediction(file)
        image = Image.open("predictions/inference/" + fileName).convert("RGB")
        display = ImageTk.PhotoImage(image)
        label = tk.Label(window, image=display)
        label.image = display
        label.grid(row=0, column=3)
        
def runPrediction(filePath):
    currModel = YOLO("runs/detect/train23/weights/best.pt")
    results = currModel(filePath,save=True, project="predictions", name="inference", exist_ok=True)
    
def createImageFrame():
    tk.Frame()
    
def getFileNameFromPath(path):
    return os.path.basename(path).split('/')[-1]
    
button = tk.Button(window,
                   text = "Donwload",
                   command= partial(handle_button)).grid(row=1, column=0)

window.mainloop()