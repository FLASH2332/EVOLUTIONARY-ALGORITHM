import random
import cv2
import numpy as np
from PIL import Image
import math
import time

sourceImg = cv2.imread('Mona_Lisa.png')

numCirclesPerImage = 100
maxCircleRadius = 250

step = 10

height, width, channels = sourceImg.shape

population = []


img = np.zeros((height, width, 3), np.uint8)

print(width, height,channels)

class Circle:
    def __init__(self, x, y, r, g, b, a, rad):
        self.x = x
        self.y = y
        self.r = r
        self.g = g
        self.b = b
        self.a = a
        self.rad = rad

    def draw(self):
        cv2.circle(img,(self.x, self.y), self.rad, (self.r, self.g, self.b), -1)


def mse(circle):
   img2 = sourceImg.copy()

   cv2.circle(img2,(circle.x, circle.y), circle.rad, (circle.r, circle.g, circle.b), -1)

   h, w, channels = sourceImg.shape
   diff = cv2.subtract(sourceImg, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

def genInitialPopulation():
    for i in range(numCirclesPerImage):
        x = random.randint(0, int(width / step)) * step
        y = random.randint(0, int(height / step)) * step

        r = random.randint(0, 256)
        g = random.randint(0, 256)
        b = random.randint(0, 256)
        a = 1

        rad = random.randint(0, maxCircleRadius)

        circle = Circle(x, y, r, g, b, a, rad)

        population.append((circle,mse(circle)))

def drawPopulation():
    for circle in population:
        circle[0].draw()

def saveImage():
    sourceImage_uint8 = np.array(sourceImg).astype(np.uint8)
    img_uint8 = img.astype(np.uint8)
    final = cv2.hconcat([sourceImage_uint8, img_uint8])
    cv2.imwrite('image.png', final)
    

def maniLoop(numGenerations):
    begin = time.time()
    genInitialPopulation()
    
    for i in range(numGenerations):
        
        drawPopulation()
        saveImage()
    begin = time.time() - begin
    print("Time: ", begin)
maniLoop(1)