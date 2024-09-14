import random
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import copy
import math

target_image = cv2.imread('Eyes.png')
target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
img_shape = target_image.shape

# Configuration
num_start = 1
num_max = 256
num_add = 1
repeat_limitMax = 150
repeat_limit = 0

min_alpha_Global = 0.0
max_alpha_Global = 1.0
min_angle_Global = 0
max_angle_Global = 90
min_radius_Global = 10
max_radius_Global = 30
min_radius_initial = 60
max_radius_initial = 80

num_gen_max = 25000
target_fitness = 100.0  # Desired fitness level
baseFit = 50.0
mutation_rate = 0.6  # probability of each attribute to mutate
large_mutation_probability = 0.1  # probability of making a large mutation

class Ellipse:
    def __init__(self, id):
        self.genotype = np.random.rand(9)

        self.min_radius = min_radius_Global
        self.max_radius = max_radius_Global
        self.min_alpha = min_alpha_Global
        self.max_alpha = max_alpha_Global
        self.min_angle = min_angle_Global
        self.max_angle = max_angle_Global
        self.id = id
        self.setMinMax()
    
    """def mutate(self):
        new_ellipse = copy.deepcopy(self)

        Randomization for Evolution
        y = random.randint(0, len(new_ellipse.genotype) - 1)

        if random.random() < 0.1:
            i, j = y, random.randint(0, len(new_ellipse.genotype) - 1)
            i, j, s = (i, j, -1) if i < j else (j, i, 1)
            new_ellipse.genotype[i: j + 1] = np.roll(new_ellipse.genotype[i: j + 1], shift=s, axis=0)
            y = j

        if random.random() < 0.25:
            new_ellipse.genotype[y] = random.random()
        else:
            new_ellipse.genotype[y] += (random.random()  - 0.5) / 3
            new_ellipse.genotype[y] = np.clip(new_ellipse.genotype[y], 0, 1)

        return new_ellipse"""
    
    def mutate(self):
        # Clone the current ellipse
        new_ellipse = copy.deepcopy(self)
        
        # Mutate position and size
        if random.random() < mutation_rate:
            if random.random() < large_mutation_probability:
                new_ellipse.genotype[0], new_ellipse.genotype[1], new_ellipse.genotype[2], new_ellipse.genotype[3] = np.random.uniform(0, 1, 4)
            else:
                new_ellipse.genotype[0] = np.clip(self.genotype[0] + np.random.uniform(-0.2, 0.2), 0, 1)
                new_ellipse.genotype[1] = np.clip(self.genotype[1] + np.random.uniform(-0.2, 0.2), 0, 1)
                new_ellipse.genotype[2] = np.clip(self.genotype[2] + np.random.uniform(-0.1, 0.1), 0, 1)
                new_ellipse.genotype[3] = np.clip(self.genotype[3] + np.random.uniform(-0.1, 0.1), 0, 1)
        
        # Mutate angle and alpha
        if random.random() < mutation_rate:
            if random.random() < large_mutation_probability:
                new_ellipse.genotype[4] = random.random()
                new_ellipse.genotype[8] = random.random()
            else:
                new_ellipse.genotype[4] = np.clip(self.genotype[4] + np.random.uniform(-0.2, 0.2), 0, 1)
                new_ellipse.genotype[8] = np.clip(self.genotype[8] + np.random.uniform(-0.2, 0.2), 0, 1)
        
        # Mutate color
        if random.random() < mutation_rate:
            if random.random() < large_mutation_probability:
                new_ellipse.genotype[5],new_ellipse.genotype[6],new_ellipse.genotype[7] = np.random.uniform(0, 1, 3)
            else:
                new_ellipse.genotype[5],new_ellipse.genotype[6],new_ellipse.genotype[7] = np.clip((self.genotype[5],self.genotype[6],self.genotype[7]) + np.random.uniform(-0.1, 0.1, 3), 0, 1)
        return new_ellipse

    def setMinMax(self):
        self.min_radius = min_radius_Global+(min_radius_initial-min_radius_Global)*math.exp((-7.0/num_max)*self.id)
        self.max_radius = max_radius_Global+(max_radius_initial-max_radius_Global)*math.exp((-7.0/num_max)*self.id)

def drawEllipses(ellipses, img_shape):
    img_result = np.full(img_shape, 255, dtype=np.uint8)
    for e in ellipses:
        x = int(e.genotype[0] * img_shape[1])
        y = int(e.genotype[1] * img_shape[0])
        a = int(e.genotype[2] * (e.max_radius - e.min_radius) + e.min_radius)
        b = int(e.genotype[3] * (e.max_radius - e.min_radius) + e.min_radius)
        angle = int(e.genotype[4] * (e.max_angle - e.min_angle) + e.min_angle)
        color = (int(e.genotype[5] * 255), int(e.genotype[6] * 255), int(e.genotype[7] * 255))
        alpha = float(e.genotype[8])

        overlay = img_result.copy()
        cv2.ellipse(overlay, (x, y), (a, b), angle, 0, 360, color, -1)
        cv2.addWeighted(overlay, alpha, img_result, 1.0 - alpha, 0, img_result)
    return img_result

def drawEllipsesOnImg(ellipses, img):
    img_shape = img.shape
    img_result = img.copy()
    for e in ellipses:
        x = int(e.genotype[0] * img_shape[1])
        y = int(e.genotype[1] * img_shape[0])
        a = int(e.genotype[2] * (e.max_radius - e.min_radius) + e.min_radius)
        b = int(e.genotype[3] * (e.max_radius - e.min_radius) + e.min_radius)
        angle = int(e.genotype[4] * (e.max_angle - e.min_angle) + e.min_angle)
        color = (int(e.genotype[5] * 255), int(e.genotype[6] * 255), int(e.genotype[7] * 255))
        alpha = float(e.genotype[8])

        overlay = img_result.copy()
        cv2.ellipse(overlay, (x, y), (a, b), angle, 0, 360, color, -1)
        cv2.addWeighted(overlay, alpha, img_result, 1.0 - alpha, 0, img_result)
    return img_result

def returnBestAdditions(population, num_add, max_trials1, max_trials2, img_shape):
    population_img = drawEllipses(population, img_shape)
    best_ellipses = []
    best_fitness = 0.0

    for _ in range(max_trials1):
        ellipses = [Ellipse(i+len(population)) for i in range(num_add)]
        new_fitness = fitness(target_image, drawEllipsesOnImg(ellipses, population_img))
        if new_fitness > best_fitness:
            best_ellipses = copy.deepcopy(ellipses)
            best_fitness = new_fitness

    for _ in range(max_trials2):
        new_ellipses =  mutateEllipses(best_ellipses, 0, len(best_ellipses), 0.75)
        new_fitness = fitness(target_image, drawEllipsesOnImg(new_ellipses, population_img))
        if new_fitness > best_fitness:
            best_ellipses = copy.deepcopy(new_ellipses)
            best_fitness = new_fitness
    return best_ellipses, best_fitness

def fitness(img1, img2):
    # return ((ssim(img1, img2, multichannel=True) + (mse(img1, img2)/65025.0))/2.0)*100.0
    # return ssim(img1, img2, multichannel=True)*100.0
    return (((1.0 - (mse(img1, img2)/65025.0)) + ssim(img1,img2,multichannel=True))/2.0)*100.0

def mutateEllipses(population, start = 0, end = 0, mutation_rate = 0.5):
    new_population = []
    new_population.extend(population[:start])
    for e in population[start:end]:
        if random.random() < mutation_rate:
            new_population.append(e.mutate())
        else:
            new_population.append(e)
    new_population.extend(population[end:])
    return new_population

def writeToFile(path, population):
    with open(path, 'w') as f:
        for e in population:
            genotypeStr = ""
            for i in range(len(e.genotype)):
                genotypeStr += str(float(round(e.genotype[i], 2)))+" "
            f.write(f"{e.id}---{genotypeStr}\n")

def hill_climbing(target_image):
    global repeat_limit, baseFit

    population = []
    inp = input("Do you want to load a previous population? (y/n): ")
    if inp == 'y':
        print("Loading previous population...")
        with open('output.txt', 'r') as f:
            for line in f:
                id, genotype = line.split("---")
                genotype = np.array([float(x) for x in genotype[1:-1].split()])
                e = Ellipse(int(id))
                e.genotype = genotype
                population.append(e)
    else:
        print("Creating new population...")
        for i in range(num_start):
            population.append(Ellipse(i))


    current_image = drawEllipses(population, img_shape)
    best_fitness = fitness(target_image, current_image)
    baseFit = best_fitness
    pastFitness = 0.0
    curr_gen = 0
    
    try:
        while curr_gen < num_gen_max and len(population) < num_max and best_fitness < target_fitness:
            if (curr_gen-1) % 1000 == 0 or curr_gen == 0:
                cv2.imwrite(f'output_gen_{curr_gen}.png', current_image)
            curr_gen += 1

            start = len(population)-num_add
            end = len(population)
            if len(population) == num_start:
                start = 0
            new_population = mutateEllipses(population, start, end, 1.0)
            new_image = drawEllipses(new_population, img_shape)
            new_fitness = fitness(target_image, new_image)
            if new_fitness > best_fitness:
                population = copy.deepcopy(new_population)
                current_image = new_image.copy()
                best_fitness = new_fitness
                print(f"Mutated->Generation: {curr_gen}, Fitness: {best_fitness:.2f}, Population Size: {len(population)}")

            # new_population = mutateEllipses(population, 0, start, 0.6)
            # new_image = drawEllipses(new_population, img_shape)
            # new_fitness = fitness(target_image, new_image)
            # if new_fitness > best_fitness:
            #     population = copy.deepcopy(new_population)
            #     current_image = new_image.copy()
            #     best_fitness = new_fitness
            #     print(f"Init_Mutated->Generation: {curr_gen}, Fitness: {best_fitness:.2f}, Population Size: {len(population)}")

            if best_fitness - pastFitness < 0.01:
                if repeat_limit == repeat_limitMax:
                    repeat_limit = 0

                    best_Temp, best_fitness_Temp = returnBestAdditions(population, num_add, 100, 50, img_shape)
                    population.extend(best_Temp)
                    current_image = drawEllipses(population, img_shape)
                    best_fitness = best_fitness_Temp
                    pastFitness = best_fitness
                    print(population[-1].max_radius,population[-1].min_radius)

                    print(f"Added->Generation: {curr_gen}, Fitness: {best_fitness:.2f}, Population Size: {len(population)}")
                else:
                    repeat_limit += 1
            else:
                repeat_limit = 0
                pastFitness = best_fitness
    except KeyboardInterrupt:
        current_image = drawEllipses(population, img_shape)
        cv2.imwrite(f'output_final.png', current_image)
        writeToFile('output.txt', population)
        return current_image
    
    writeToFile('output.txt', population)
    return current_image

# Perform hill climbing
final_image = hill_climbing(target_image)

# Display the final image
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()