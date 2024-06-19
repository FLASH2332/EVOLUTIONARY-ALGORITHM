import random
import cv2
import numpy as np
from PIL import Image
import math
import time

sourceImg = cv2.imread('chuck.png')

numCirclesPerImage = 200
maxCircleRadius = 25

step = 10

height, width, channels = sourceImg.shape

population = []

mutationRate=0.7
hardMutationRate=0.1


img = np.zeros((height, width, 3), np.float16)


print(width, height,channels)

def clamp(c,a=0,b=255):
    return max(a,min(b,c))

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
       

import numpy as np

def mse(circle, sourceImg):
    # Create an empty mask with the same dimensions as sourceImg
    mask = np.zeros_like(sourceImg)
    
    # Draw the circle on the mask instead of the image copy
    cv2.circle(mask, (circle.x, circle.y), circle.rad, (circle.r, circle.g, circle.b), -1)
    
    # Calculate the difference only where the mask is not zero
    diff = sourceImg[mask != 0] - mask[mask != 0]
    
    # Calculate MSE using efficient NumPy operations
    err = np.sum(diff ** 2)
    mse = err   # Normalize by the number of changed pixels
    
    return mse

def rankCircles(population):
    return sorted(population, key=lambda x: x[1])

def mutate(circle):
    x = circle.x
    y = circle.y
    r = circle.r
    g = circle.g
    b = circle.b
    a = circle.a
    rad = circle.rad
    probability= random.uniform(0.0,1.0)

    if(probability<hardMutationRate):  
        x = random.randint(0, int(width / step)) * step
        y = random.randint(0, int(height / step)) * step

        r = random.randint(0, 256)
        g = random.randint(0, 256)
        b = random.randint(0, 256)
        a = 0.5

        rad = random.randint(0, maxCircleRadius)
    elif(probability<mutationRate):
        propbality1=random.uniform(0.0,1.0)
        if(propbality1<0.5):
            x =clamp(x+ random.randint(-step, step),0,width)
            y =clamp(y+random.randint(-step, step),0,height)
            rad =clamp(rad+random.randint(-5, 5),0,maxCircleRadius)
        else:
            a += random.randint(-10, 10)
            r=clamp(r+ random.randint(-10, 10))
            g=clamp(g+ random.randint(-10, 10))
            b=clamp(b+ random.randint(-10, 10))
            a=clamp(a+ random.uniform(-0.1,0.1),0.0,1.0)

    return Circle(x, y, r, g, b, a, rad)


def genInitialPopulation():
    for i in range(numCirclesPerImage):
        x = random.randint(0, int(width / step)) * step
        y = random.randint(0, int(height / step)) * step

        r = random.randint(0, 256)
        g = random.randint(0, 256)
        b = random.randint(0, 256)
        a = 0.5

        rad = random.randint(0, maxCircleRadius)

        circle = Circle(x, y, r, g, b, a, rad)

        population.append((circle,mse(circle,sourceImg)))
        circle.draw()

def drawPopulation():
    for circle in population:
        circle[0].draw()

def saveImage():
    # sourceImage_uint8 = np.array(sourceImg).astype(np.uint8)
    img_uint8 = img.astype(np.uint8)
    # final = cv2.hconcat([sourceImage_uint8, img_uint8])
    cv2.imwrite('image.png', img)
    

def maniLoop(numGenerations):
    begin = time.time()
    genInitialPopulation()
    
    for i in range(numGenerations):
        for j in range(len(population)):
            mutatedCircle = mutate(population[j][0])
            population[j] = (mutatedCircle, mse(mutatedCircle,sourceImg))
        saveImage()
    begin = time.time() - begin
    print("Time: ", begin)
maniLoop(5)
