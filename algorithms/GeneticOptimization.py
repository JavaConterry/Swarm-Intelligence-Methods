import numpy as np
import random
import matplotlib.pyplot as plt
import math as m

class Generation:
    def __init__(self, population, fitness_fn, bits, dimentions):
        self.population = population  # Population consists of tuples of binary lists
        self.fitness_fn = fitness_fn
        self.fitness_scores = np.array([fitness_fn(ind) for ind in population])
        self.bits = bits
        self.dimentions = dimentions

    def update_fitness_scores(self):
        self.fitness_scores = np.array([self.fitness_fn(ind) for ind in self.population])

    def select(self):  # Roulette selection
        shift = abs(min(self.fitness_scores)) + 1e-6 if min(self.fitness_scores) < 0 else 0
        total_fitness = sum([(val + shift) for val in self.fitness_scores])
        cumulative_probab = np.cumsum([(val + shift) / total_fitness for val in self.fitness_scores])
        
        selected_population = []
        for _ in range(len(self.population)):
            r = random.random()
            for i, cs in enumerate(cumulative_probab):
                if r <= cs:
                    selected_population.append(self.population[i])
                    break
        return Generation(selected_population, self.fitness_fn, self.bits, self.dimentions)
    
    def crossover(self):  # Exchange second half

        if self.dimentions == 2:
            for i in range(0, len(self.population)-1, 2):
                ch1 = self.population[i][:self.bits//2]
                ch1.extend(self.population[i+1][self.bits//2:])
                ch2 = self.population[i+1][:self.bits//2]
                ch2.extend(self.population[i][self.bits//2:])
                self.population[i:i+2] = [ch1, ch2]
            self.update_fitness_scores()
        elif self.dimentions == 3:
            for i in range(0, len(self.population) - 1, 2):
                ch1_x = self.population[i][0][:self.bits//2] + self.population[i+1][0][self.bits//2:]
                ch1_y = self.population[i][1][:self.bits//2] + self.population[i+1][1][self.bits//2:]
                ch2_x = self.population[i+1][0][:self.bits//2] + self.population[i][0][self.bits//2:]
                ch2_y = self.population[i+1][1][:self.bits//2] + self.population[i][1][self.bits//2:]
                self.population[i:i+2] = [(ch1_x, ch1_y), (ch2_x, ch2_y)]
            self.update_fitness_scores()
        else:
            print(f'Number of dimentions {self.dimentions} is not supported, mutation is skipped')


    def mutate(self, random_factor=0.1):
        if self.dimentions == 2:
            for i, ind in enumerate(self.population):
                for j in range(len(ind)):
                    r = random.random()
                    if r<=random_factor:
                        ind[j] = str(1-int(ind[j]))
                        self.population[i] = ind
        elif self.dimentions == 3:
            for i, (ind_x, ind_y) in enumerate(self.population):
                for j in range(len(ind_x)):
                    if random.random() <= random_factor:
                        ind_x[j] = str(1 - int(ind_x[j]))
                    if random.random() <= random_factor:
                        ind_y[j] = str(1 - int(ind_y[j]))
                self.population[i] = (ind_x, ind_y)
        else:
            print(f'Number of dimentions {self.dimentions} is not supported, mutation is skipped')

    def next_generation(self):
        selected = self.select()
        selected.crossover()
        selected.mutate()
        return selected

def bin_list(num, bits):
    return [char for char in f'{num:0{bits}b}']

def to_10b(x):
    return np.sum([int(a) * 2 ** (len(x) - i - 1) for i, a in enumerate(x)])

def main3d_example():
    # def optimisation_fn_3d(ind):
    #     x, y = to_10b(ind[0]), to_10b(ind[1])
    #     return -1 * ((x - 5) ** 2 + (y - 5) ** 2 - 10 * np.sin(x) * np.cos(y))
    
    def optimisation_fn_3d(ind): # Shuffer function
        x, y = to_10b(ind[0]), to_10b(ind[1])
        return 1/2 + (m.sin(x**2+ y**2) - 1/2)/(1+0.001*(x**2+y**2))**2
    
    bits = 5  # Increase bits for more precise representation
    population = [(bin_list(random.randint(0, 2**bits - 1), bits),
                   bin_list(random.randint(0, 2**bits - 1), bits)) for _ in range(10)]
    generation = Generation(population, optimisation_fn_3d, bits, dimentions=3)
    
    x_vals = np.linspace(0, 2 ** bits - 1, 50)
    y_vals = np.linspace(0, 2 ** bits - 1, 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([[optimisation_fn_3d((bin_list(int(x), bits), bin_list(int(y), bits))) for x in x_vals] for y in y_vals])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    scatter = ax.scatter([], [], [], color='red', label='Population')
    ax.legend()
    
    prev = np.max(generation.fitness_scores)
    times = 0
    n = 0
    
    while n <= 100 or times <= 3:
        population_x = [to_10b(ind[0]) for ind in generation.population]
        population_y = [to_10b(ind[1]) for ind in generation.population]
        population_z = [optimisation_fn_3d(ind) for ind in generation.population]
        scatter._offsets3d = (population_x, population_y, population_z)
        # plt.pause(0.001)
        ax.set_title(f'Generation N: {n}')
        
        generation = generation.next_generation()
        
        if prev == np.max(generation.fitness_scores):
            times += 1
        else:
            prev = np.max(generation.fitness_scores)
            times = 0
        
        n += 1
    
    print('\nOptimal value found:', -1 * np.max(generation.fitness_scores))
    plt.show()

def main2d_example():
    def optimisation_fn(x):
        x = to_10b(x)
        return -1 * (5 - 24*x + 17*x**2 - x**3 * 11/3 + (x**4)/4)
    def vis_func(x):
        return -1 * (5 - 24*x + 17*x**2 - x**3 * 11/3 + (x**4)/4)
    
    bits = 3
    generation = Generation([bin_list(int(num), bits) for num in range(0, 2**bits, 2)], optimisation_fn, bits, dimentions=2)
    x_vals = np.linspace(0, 2**bits - 1, 2**(bits+2))
    y_vals = [vis_func(x) for x in x_vals]
    
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label='Function')
    scatter = ax.scatter([], [], color='red', label='Population')
    ax.legend()
    
    prev = np.max(generation.fitness_scores); times = 0
    n=0
    while n<=100 or times <= 3:
        population_x = [to_10b(ind) for ind in generation.population]
        population_y = [optimisation_fn(ind) for ind in generation.population]
        scatter.set_offsets(np.c_[population_x, population_y])
        # plt.pause(0.1)
        plt.title(f'Generation N: {n}')

        generation = generation.next_generation()

        if prev == np.max(generation.fitness_scores):
            times+=1
        else:
            prev = np.max(generation.fitness_scores)
            times = 0

        n+=1

    print('\nOptimal value found:', -1 *np.max(generation.fitness_scores))

    plt.ioff()
    plt.show()

# main2d_example()
main3d_example()