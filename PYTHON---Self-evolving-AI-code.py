import random
import numpy as np

# Objective function to optimize (e.g., maximize this function)
def fitness_function(x):
    return -x**2 + 4*x + 10  # Parabolic function

# Create an initial population of individuals
def initialize_population(size, lower_bound, upper_bound):
    return [random.uniform(lower_bound, upper_bound) for _ in range(size)]

# Evaluate fitness of individuals
def evaluate_population(population):
    return [fitness_function(individual) for individual in population]

# Select parents based on fitness (roulette wheel selection)
def select_parents(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    return np.random.choice(population, size=2, p=probabilities)

# Perform crossover between two parents
def crossover(parent1, parent2):
    alpha = random.uniform(0, 1)
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2

# Mutate an individual slightly
def mutate(individual, mutation_rate, lower_bound, upper_bound):
    if random.random() < mutation_rate:
        mutation = random.uniform(-1, 1)
        individual += mutation
        # Keep within bounds
        individual = max(lower_bound, min(upper_bound, individual))
    return individual

# Genetic Algorithm (self-evolving loop)
def genetic_algorithm(generations, population_size, lower_bound, upper_bound, mutation_rate):
    # Initialize population
    population = initialize_population(population_size, lower_bound, upper_bound)
    best_individual = None
    best_fitness = float('-inf')

    for generation in range(generations):
        # Evaluate fitness of population
        fitness = evaluate_population(population)
        
        # Track the best individual
        current_best_fitness = max(fitness)
        current_best_individual = population[fitness.index(current_best_fitness)]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual

        print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}, Best Individual = {best_individual:.2f}")

        # Select new population through crossover and mutation
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, lower_bound, upper_bound)
            child2 = mutate(child2, mutation_rate, lower_bound, upper_bound)
            new_population.extend([child1, child2])

        population = new_population

    return best_individual, best_fitness

# Parameters for the algorithm
GENERATIONS = 50
POPULATION_SIZE = 20
LOWER_BOUND = -10
UPPER_BOUND = 10
MUTATION_RATE = 0.1

# Run the genetic algorithm
best_solution, best_solution_fitness = genetic_algorithm(
    GENERATIONS, POPULATION_SIZE, LOWER_BOUND, UPPER_BOUND, MUTATION_RATE
)

print(f"\nFinal Best Solution: x = {best_solution:.2f}, Fitness = {best_solution_fitness:.2f}")
