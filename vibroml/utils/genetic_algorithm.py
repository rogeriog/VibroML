# vibroml/utils/genetic_algorithm.py

import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self,
                 population_size,
                 mutation_rate,
                 displacement_scale_bounds, # [min, max] for mode1 displacement
                 ratio_mode2_to_mode1_bounds, # [min, max] for ratio
                 cell_scale_bounds, # [min, max] for a,b,c percentage change
                 cell_angle_bounds, # [min, max] for alpha,beta,gamma degree change
                 num_offspring=30, # Number of new individuals to generate per evolution step
                 selection_strategy='tournament', # or 'roulette', 'rank'
                 tournament_size=3 # For tournament selection
                ):
        """
        Initializes the Genetic Algorithm.

        Args:
            population_size (int): The number of individuals in the population.
            mutation_rate (float): The probability of a gene mutating (0.0 to 1.0).
            displacement_scale_bounds (tuple): (min, max) for displacement_scale_mode1.
            ratio_mode2_to_mode1_bounds (tuple): (min, max) for ratio_mode2_to_mode1.
            cell_scale_bounds (tuple): (min, max) for percentage change in a, b, c.
            cell_angle_bounds (tuple): (min, max) for degree change in alpha, beta, gamma.
            num_offspring (int): Number of new individuals to generate in each evolve step.
            selection_strategy (str): Method for selecting parents ('tournament', 'roulette', 'rank').
            tournament_size (int): Number of individuals in each tournament if using 'tournament' selection.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.displacement_scale_bounds = displacement_scale_bounds
        self.ratio_mode2_to_mode1_bounds = ratio_mode2_to_mode1_bounds
        self.cell_scale_bounds = cell_scale_bounds
        self.cell_angle_bounds = cell_angle_bounds
        self.num_offspring = num_offspring
        self.selection_strategy = selection_strategy
        self.tournament_size = tournament_size

        # The population will be a list of dictionaries:
        # [{'params': (disp_scale, ratio, (sa,sb,sc,sal,sbe,sga)), 'fitness': energy}, ...]
        self.population = []

    def _generate_random_individual(self):
        """Generates a single random individual (parameter set) within bounds."""
        disp_scale = random.uniform(*self.displacement_scale_bounds)
        ratio = random.uniform(*self.ratio_mode2_to_mode1_bounds)
        cell_scales = [random.uniform(*self.cell_scale_bounds) for _ in range(3)]
        cell_angles = [random.uniform(*self.cell_angle_bounds) for _ in range(3)]
        cell_transformation_vector = tuple(cell_scales + cell_angles)
        return (disp_scale, ratio, cell_transformation_vector)

    def initialize_population(self, initial_individuals=None):
        """
        Initializes the population.
        If initial_individuals are provided, they form the starting population.
        Otherwise, a random population is generated.
        """
        self.population = []
        if initial_individuals:
            # Initial individuals should be in the format:
            # [{'params': (disp_scale, ratio, cell_transform_vec), 'fitness': energy}, ...]
            self.population.extend(initial_individuals)
            # If initial_individuals are fewer than population_size, fill the rest randomly
            while len(self.population) < self.population_size:
                self.population.append({'params': self._generate_random_individual(), 'fitness': None})
        else:
            for _ in range(self.population_size):
                self.population.append({'params': self._generate_random_individual(), 'fitness': None})

        print(f"Initialized GA population with {len(self.population)} individuals.")

    def _select_parents(self):
        """Selects two parents from the current population based on the selection strategy."""
        if not self.population or all(ind['fitness'] is None for ind in self.population):
            raise ValueError("Population is empty or no fitness values available for selection.")

        # Filter out individuals with None fitness for selection
        eligible_population = [ind for ind in self.population if ind['fitness'] is not None]
        if not eligible_population:
            # If no individuals have fitness, fall back to random selection or re-initialization
            print("Warning: No individuals with valid fitness for selection. Selecting randomly.")
            return random.sample(self.population, 2) # Select any two, even if fitness is None

        if self.selection_strategy == 'tournament':
            # Tournament selection: pick N random individuals, choose the best one
            def tournament_selection():
                contenders = random.sample(eligible_population, min(self.tournament_size, len(eligible_population)))
                # Fitness is energy, so lower is better (min energy)
                return min(contenders, key=lambda x: x['fitness'])

            parent1 = tournament_selection()
            parent2 = tournament_selection()
            # Ensure parents are distinct if possible, though not strictly required for GA
            while parent1 == parent2 and len(eligible_population) > 1:
                parent2 = tournament_selection()
            return parent1['params'], parent2['params']

        elif self.selection_strategy == 'roulette':
            # Roulette wheel selection: probability proportional to fitness (inverted for minimization)
            # Convert energy (minimization) to a "score" (maximization)
            # A common way is max_energy - current_energy + small_constant
            fitness_values = [ind['fitness'] for ind in eligible_population]
            max_energy = max(fitness_values)
            # Add a small constant to avoid zero or negative scores if max_energy == current_energy
            scores = [max_energy - f + 1e-6 for f in fitness_values]
            total_score = sum(scores)
            if total_score == 0: # All scores are effectively zero, pick randomly
                print("Warning: All fitness scores are effectively zero. Selecting randomly.")
                return random.sample(eligible_population, 2)[0]['params'], random.sample(eligible_population, 2)[1]['params']

            probabilities = [s / total_score for s in scores]
            
            # Select two parents based on probabilities
            parent1_idx = np.random.choice(len(eligible_population), p=probabilities)
            parent2_idx = np.random.choice(len(eligible_population), p=probabilities)
            
            # Ensure parents are distinct if possible
            while parent1_idx == parent2_idx and len(eligible_population) > 1:
                parent2_idx = np.random.choice(len(eligible_population), p=probabilities)
            
            return eligible_population[parent1_idx]['params'], eligible_population[parent2_idx]['params']

        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def _crossover(self, parent1_params, parent2_params):
        """Performs single-point crossover between two parent parameter sets."""
        # Parameters are: (disp_scale, ratio, (sa,sb,sc,sal,sbe,sga))
        # Total 1 + 1 + 6 = 8 genes
        
        # Convert tuples to lists for mutability
        p1_list = [parent1_params[0], parent1_params[1]] + list(parent1_params[2])
        p2_list = [parent2_params[0], parent2_params[1]] + list(parent2_params[2])

        crossover_point = random.randint(1, len(p1_list) - 1) # Crossover point can be after any gene except the last

        child1_list = p1_list[:crossover_point] + p2_list[crossover_point:]
        child2_list = p2_list[:crossover_point] + p1_list[crossover_point:]

        # Convert back to original structure
        child1_params = (child1_list[0], child1_list[1], tuple(child1_list[2:]))
        child2_params = (child2_list[0], child2_list[1], tuple(child2_list[2:]))

        return child1_params, child2_params

    def _mutate(self, individual_params):
        """Mutates an individual's parameter set based on mutation_rate."""
        # Parameters are: (disp_scale, ratio, (sa,sb,sc,sal,sbe,sga))
        mutated_params_list = [individual_params[0], individual_params[1]] + list(individual_params[2])

        # Define bounds for each gene type
        bounds = [self.displacement_scale_bounds, self.ratio_mode2_to_mode1_bounds] + \
                 [self.cell_scale_bounds] * 3 + [self.cell_angle_bounds] * 3

        for i in range(len(mutated_params_list)):
            if random.random() < self.mutation_rate:
                min_val, max_val = bounds[i]
                # Apply a small random perturbation within bounds
                # A common mutation strategy is to add a small Gaussian noise
                # or simply re-randomize within a smaller range around the current value
                # For "high mutation", re-randomizing within the full bounds is also an option.
                # Let's try re-randomizing within the full bounds for high mutation.
                mutated_params_list[i] = random.uniform(min_val, max_val)
                
                # Ensure bounds are respected after mutation
                mutated_params_list[i] = max(min_val, min(max_val, mutated_params_list[i]))

        return (mutated_params_list[0], mutated_params_list[1], tuple(mutated_params_list[2:]))

    def evolve(self, current_population_with_fitness):
        """
        Evolves the population for one generation.
        
        Args:
            current_population_with_fitness (list): A list of dictionaries,
                each with 'params' (tuple) and 'fitness' (float, energy).
                This is the result of the previous iteration's calculations.
        
        Returns:
            list: A list of new parameter sets (individuals) for the next generation.
                  These are tuples: (disp_scale, ratio, cell_transformation_vector).
        """
        # Update the GA's internal population with the evaluated fitness values
        self.population = current_population_with_fitness
        
        # Sort population by fitness (energy), lowest energy is best
        self.population.sort(key=lambda x: x['fitness'])

        new_offspring_params = []
        
        # Elitism: Keep the best individual(s) directly
        # Let's keep the single best individual without modification
        if self.population:
            new_offspring_params.append(self.population[0]['params'])
            # Adjust num_offspring if we're using elitism to ensure total count is met
            num_to_generate = self.num_offspring - 1
        else:
            num_to_generate = self.num_offspring

        # Generate new offspring until num_offspring is met
        while len(new_offspring_params) < self.num_offspring:
            try:
                parent1_params, parent2_params = self._select_parents()
            except ValueError as e:
                print(f"Error during parent selection: {e}. Generating random individuals instead.")
                # Fallback: if selection fails, generate random individuals
                new_offspring_params.append(self._generate_random_individual())
                continue # Skip to next iteration of while loop

            child1_params, child2_params = self._crossover(parent1_params, parent2_params)

            mutated_child1 = self._mutate(child1_params)
            mutated_child2 = self._mutate(child2_params)

            new_offspring_params.append(mutated_child1)
            if len(new_offspring_params) < self.num_offspring:
                new_offspring_params.append(mutated_child2)
        
        print(f"Generated {len(new_offspring_params)} new offspring for the next generation.")
        return new_offspring_params