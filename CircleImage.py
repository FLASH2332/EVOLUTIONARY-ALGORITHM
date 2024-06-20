import random
import cv2
import numpy as np
from PIL import Image
import math
import time

sourceImg = cv2.imread('chuck.png')

numCirclesPerImage = 250
maxCircleRadius = 30
minCircleRadius = 5

step = 5

height, width, channels = sourceImg.shape

population = []

mutationRate=0.4
hardMutationRate=0.1
matingProbability=0.9
passingRate=0.3


img = np.zeros((height, width, 3), np.uint8)


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
        img2 = img.copy()
        cv2.circle(img2,(self.x,self.y),self.rad,(self.r,self.g,self.b), -1)
        cv2.addWeighted(img, 1.0-0.6, img2, 0.6, 0, img)
       

import numpy as np

def mse(circle, sourceImg):
    img2 = img.copy()
    cv2.circle(img2,(circle.x,circle.y),circle.rad,(circle.r,circle.g,circle.b), -1)
    difference = sourceImg - img2
    mse = np.sum(difference**2)
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
        a = random.uniform(0.0, 1.0)

        rad = random.randint(minCircleRadius, maxCircleRadius)
    elif(probability<mutationRate):
        propbality1=random.uniform(0.0,1.0)
        if(propbality1<0.2):
            x =clamp(x+ random.randint(-step, step),0,width)
            y =clamp(y+random.randint(-step, step),0,height)
            rad =clamp(rad+random.randint(-5, 5),minCircleRadius,maxCircleRadius)
        else:
            r=clamp(r+ random.randint(-10, 10))
            g=clamp(g+ random.randint(-10, 10))
            b=clamp(b+ random.randint(-10, 10))
            a=clamp(a+ random.uniform(-0.1,0.1),0.0,1.0)

    return Circle(x, y, r, g, b, a, rad)

def mate(circle1, circle2):
    change=random.uniform(0.0,1.0)
    list_circle_1=[circle1.x,circle1.y,circle1.r,circle1.g,circle1.b,circle1.a,circle1.rad]
    list_circle_2=[circle2.x,circle2.y,circle2.r,circle2.g,circle2.b,circle2.a,circle2.rad]
    list_child=[]
    list_child2=[]
    if(change>=matingProbability):
        parameters=random.randint(1,8)
        list_child=list_circle_1[:parameters]+list_circle_2[parameters:]
        list_child2=list_circle_2[:parameters]+list_circle_1[parameters:]
        return [Circle(list_child[0],list_child[1],list_child[2],list_child[3],list_child[4],list_child[5],list_child[6]),
                Circle(list_child2[0],list_child2[1],list_child2[2],list_child2[3],list_child2[4],list_child2[5],list_child2[6])]
    else:
        return [circle1,circle2]

def mse_total(population):
    difference=sourceImg-img
    mse=np.sum(difference**2)
    return mse
def genInitialPopulation():
    for i in range(numCirclesPerImage):
        x = random.randint(0, int(width / step)) * step
        y = random.randint(0, int(height / step)) * step

        r = random.randint(0, 256)
        g = random.randint(0, 256)
        b = random.randint(0, 256)
        a = random.uniform(0.0, 1.0)

        rad = random.randint(minCircleRadius, maxCircleRadius)

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
    global population
    global img
    begin = time.time()
    genInitialPopulation()
    
    for i in range(numGenerations):
        newPopulation=[]
        population=rankCircles(population)
        if(i%10==0):
            print("Generation: ", i)
            print("MSE: ",mse_total(population))
        for j in range(len(population)):
            if(j<len(population)*passingRate):
                mutatedCircle = mutate(population[j][0])
                newPopulation.append((mutatedCircle,mse(mutatedCircle,sourceImg)))
            else:
                k=random.randint(0,len(population)-1)
                matedCircles=mate(population[k-1][0],population[k][0])
                # changes can be made here using 2 random numbers instead of continuous parents
                mutate1=mutate(matedCircles[0])
                mutate2=mutate(matedCircles[1])
                newPopulation.append((mutate1,mse(mutate1,sourceImg)))
                # newPopulation.append((mutate2,mse(mutate2,sourceImg)))
        population=newPopulation
        img = np.zeros((height, width, 3), np.uint8)
        drawPopulation()
    saveImage()
    begin = time.time() - begin
    print("Time: ", begin)
    print(len(population))
maniLoop(500)
