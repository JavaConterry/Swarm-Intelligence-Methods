import numpy as np
import random
import matplotlib.pyplot as plt

class Generation:
    def __init__(self, population, fitness_fn, bits):
        self.population = population
        self.fitness_fn = fitness_fn
        self.fitness_scores = np.array([fitness_fn(ind) for ind in population])
        self.bits = bits

    def update_fitness_scores(self):
        self.fitness_scores = np.array([self.fitness_fn(ind) for ind in self.population])

    def select(self): # roulette rule
        shift = abs(min(self.fitness_scores))+ 1e-6 if min(self.fitness_scores)< 0 else 0; 
        total_fitness = sum([(val + shift) for val in self.fitness_scores]); 

        cumulative_probab = np.cumsum([(val+shift)/total_fitness for val in self.fitness_scores])

        selected_population = []
        for _ in range(len(self.population)):
            r = random.random()
            for i, cs in enumerate(cumulative_probab):
                if r<=cs:
                    selected_population.append(self.population[i])
                    break

        return Generation(selected_population, self.fitness_fn, self.bits)
   
    def crossover(self): # exchange second half
        for i in range(0, len(self.population)-1, 2):
            ch1 = self.population[i][:self.bits//2]
            ch1.extend(self.population[i+1][self.bits//2:])
            ch2 = self.population[i+1][:self.bits//2]
            ch2.extend(self.population[i][self.bits//2:])
            self.population[i:i+2] = [ch1, ch2]
        self.update_fitness_scores()

    def mutate(self, random_factor = 0.2):
        for i, ind in enumerate(self.population):
            for j in range(len(ind)):
                r = random.random()
                if r<=random_factor:
                    ind[j] = str(1-int(ind[j]))
                    self.population[i] = ind

    def next_generation(self):
        selected = self.select()
        selected.crossover()
        selected.mutate()
        
        return selected


def bin_list(num, bits):
    # return [char for char in bin(num)[2:]]
    return [char for char in f'{num:0{bits}b}']

def to_10b(x):
    a = [int(a)*2**(len(x)-i-1) for i,a in enumerate(x)]
    return np.sum(a)

def main():
    def optimisation_fn(x):
        x = to_10b(x)
        return -1 * (5 - 24*x + 17*x**2 - x**3 * 11/3 + (x**4)/4)
    def vis_func(x):
        return -1 * (5 - 24*x + 17*x**2 - x**3 * 11/3 + (x**4)/4)
    
    bits = 3
    generation = Generation([bin_list(int(num), bits) for num in range(0, 2**bits, 2)], optimisation_fn, bits)
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
        plt.pause(0.1)
        plt.title(f'Generation N: {n}')
        generation = generation.next_generation()
        if prev == np.max(generation.fitness_scores):
            times+=1
        else:
            prev = np.max(generation.fitness_scores)
            times = 0
        n+=1

    print('\n', -1 *np.max(generation.fitness_scores))

    plt.ioff()
    plt.show()

main()


    