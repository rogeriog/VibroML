# vibroml/utils/genetic_algorithm.py

import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self,
                 population_size,
                 mutation_rate,
                 displacement_scale_bounds,
                 ratio_mode2_to_mode1_bounds,
                 cell_scale_bounds,
                 cell_angle_bounds,
                 supercell_variants, 
                 phase_factor_off_ratio=0.5,
                 num_offspring=30,
                 selection_strategy='tournament',
                 tournament_size=10
                ):
        """
        Initializes the Genetic Algorithm.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.displacement_scale_bounds = displacement_scale_bounds
        self.ratio_mode2_to_mode1_bounds = ratio_mode2_to_mode1_bounds
        self.cell_scale_bounds = cell_scale_bounds
        self.cell_angle_bounds = cell_angle_bounds
        self.supercell_variants = supercell_variants
        self.num_offspring = num_offspring
        self.selection_strategy = selection_strategy
        self.tournament_size = tournament_size
        self.phase_factor_off_ratio = phase_factor_off_ratio
        # The population will be a list of dictionaries:
        # [{'params': (disp_scale, ratio, cell_transform_vec, supercell_variant), 'fitness': energy}, ...]
        self.population = []

    def _generate_random_individual(self):
        """Generates a single random individual (parameter set) within bounds."""
        disp_scale = random.uniform(*self.displacement_scale_bounds)
        ratio = random.uniform(*self.ratio_mode2_to_mode1_bounds)
        cell_scales = [random.uniform(*self.cell_scale_bounds) for _ in range(3)]
        cell_angles = [random.uniform(*self.cell_angle_bounds) for _ in range(3)]
        cell_transformation_vector = tuple(cell_scales + cell_angles)
        
        # Pick a random supercell variant from the provided list
        supercell_variant = random.choice(self.supercell_variants)
        use_phase_factor = random.choices([True, False], weights=[1-self.phase_factor_off_ratio, self.phase_factor_off_ratio])[0]

        return (disp_scale, ratio, cell_transformation_vector, supercell_variant, use_phase_factor)

    def initialize_population(self, initial_individuals=None):
        """
        Initializes the population.
        If initial_individuals are provided, they form the starting population.
        Otherwise, a random population is generated.
        """
        self.population = []
        if initial_individuals:
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
        eligible_population = [ind for ind in self.population if ind['fitness'] is not None]  
  
        if not eligible_population:  
            # If no individuals have valid fitness, we cannot select based on fitness.  
            # This should ideally not happen if initialize_population is called correctly  
            # and fitness is evaluated for at least some individuals.  
            # Fallback: Generate random individuals if no valid population to select from.  
            print("Warning: No individuals with valid fitness for selection. Generating random parents.")  
            return self._generate_random_individual(), self._generate_random_individual()  
  
        if len(eligible_population) < 2:  
            # If there's only one or zero eligible individuals, we can't pick two distinct parents.  
            # In this case, we might pick the same parent twice, or generate a random one.  
            print(f"Warning: Only {len(eligible_population)} individual(s) with valid fitness. Parent selection might be limited.")  
            if len(eligible_population) == 1:  
                # If only one, pick it twice  
                return eligible_population[0]['params'], eligible_population[0]['params']  
            else: # len == 0, handled above  
                return self._generate_random_individual(), self._generate_random_individual()  
  
  
        if self.selection_strategy == 'tournament':  
            def tournament_selection_single():  
                # Ensure tournament_size doesn't exceed eligible_population size  
                current_tournament_size = min(self.tournament_size, len(eligible_population))  
                if current_tournament_size == 0: # Should not happen due to checks above  
                    raise ValueError("Cannot perform tournament selection with no eligible individuals.")  
                  
                contenders = random.sample(eligible_population, current_tournament_size)  
                return min(contenders, key=lambda x: x['fitness'])  
  
            parent1_obj = tournament_selection_single()  
            parent2_obj = tournament_selection_single()  
  
            # Ensure parent2 is different from parent1 if possible  
            # This loop ensures we get two *different* individuals (objects) if the population allows.  
            # If eligible_population has only one unique object, this loop will eventually break.  
            attempts = 0  
            max_attempts = 10 # Prevent infinite loop for very small populations  
            while parent1_obj == parent2_obj and len(eligible_population) > 1 and attempts < max_attempts:  
                parent2_obj = tournament_selection_single()  
                attempts += 1  
              
            # If after max_attempts, they are still the same, and population > 1,  
            # it means tournament_selection_single is consistently picking the same object.  
            # This is unlikely with random.sample unless eligible_population is tiny.  
            # In such a case, it's acceptable to proceed with identical parents.  
  
            return parent1_obj['params'], parent2_obj['params']  
  
        elif self.selection_strategy == 'roulette':  
            # ... (your existing roulette code, which seems fine) ...  
            fitness_values = [ind['fitness'] for ind in eligible_population]  
            max_energy = max(fitness_values)  
            scores = [max_energy - f + 1e-6 for f in fitness_values]  
            total_score = sum(scores)  
            if total_score == 0:  
                print("Warning: All fitness scores are effectively zero. Selecting randomly.")  
                p1, p2 = random.sample(eligible_population, 2)  
                return p1['params'], p2['params']  
  
            probabilities = [s / total_score for s in scores]  
            # Ensure replace=True for small populations, as we might pick the same parent twice  
            parent_indices = np.random.choice(len(eligible_population), size=2, p=probabilities, replace=True)  
            return eligible_population[parent_indices[0]]['params'], eligible_population[parent_indices[1]]['params']  
  
        else:  
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def _crossover(self, parent1_params, parent2_params):
        """Performs crossover on numerical genes and categorical (supercell) genes separately."""
        # parent_params = (disp_scale, ratio, cell_transform_vec, supercell_variant)
        
        # 1. Crossover for numerical parts (genes 0, 1, and the tuple at 2)
        p1_numeric = [parent1_params[0], parent1_params[1]] + list(parent1_params[2])
        p2_numeric = [parent2_params[0], parent2_params[1]] + list(parent2_params[2])
        
        crossover_point = random.randint(1, len(p1_numeric) - 1)
        
        child1_numeric_list = p1_numeric[:crossover_point] + p2_numeric[crossover_point:]
        child2_numeric_list = p2_numeric[:crossover_point] + p1_numeric[crossover_point:]
        
        # 2. Crossover for the supercell gene (categorical)
        # Each child randomly inherits the supercell from one of the parents.
        child1_sc = random.choice([parent1_params[3], parent2_params[3]])
        child2_sc = random.choice([parent1_params[3], parent2_params[3]])
        
        # 3. Crossover for the use_phase_factor gene (boolean)  
        child1_pf = random.choice([parent1_params[4], parent2_params[4]])  
        child2_pf = random.choice([parent1_params[4], parent2_params[4]])

        # 3. Reconstruct the full parameter tuples for the children
        child1_params = (child1_numeric_list[0], child1_numeric_list[1], tuple(child1_numeric_list[2:]), child1_sc, child1_pf)
        child2_params = (child2_numeric_list[0], child2_numeric_list[1], tuple(child2_numeric_list[2:]), child2_sc, child2_pf)
        
        return child1_params, child2_params

    def _mutate(self, individual_params):
        """Mutates numerical and categorical genes of an individual."""
        # individual_params = (disp_scale, ratio, cell_transform_vec, supercell_variant)
        disp_scale, ratio, cell_transform_vec, supercell_variant, use_phase_factor = individual_params
        
        # 1. Mutate numerical parts
        mutated_numeric_list = [disp_scale, ratio] + list(cell_transform_vec)
        bounds = [self.displacement_scale_bounds, self.ratio_mode2_to_mode1_bounds] + \
                 [self.cell_scale_bounds] * 3 + [self.cell_angle_bounds] * 3
                 
        for i in range(len(mutated_numeric_list)):
            if random.random() < self.mutation_rate:
                min_val, max_val = bounds[i]
                mutated_numeric_list[i] = random.uniform(min_val, max_val)
                mutated_numeric_list[i] = max(min_val, min(max_val, mutated_numeric_list[i]))
                
        mutated_disp_scale = mutated_numeric_list[0]
        mutated_ratio = mutated_numeric_list[1]
        mutated_cell_transform_vec = tuple(mutated_numeric_list[2:])
        
        # 2. Mutate the supercell gene
        mutated_supercell_variant = supercell_variant
        if random.random() < self.mutation_rate and len(self.supercell_variants) > 1:
            # Pick a new, different supercell variant from the list of possibilities
            possible_new_variants = [sc for sc in self.supercell_variants if sc != supercell_variant]
            if possible_new_variants:
                mutated_supercell_variant = random.choice(possible_new_variants)

        # 3. Mutate the use_phase_factor gene  
        mutated_use_phase_factor = use_phase_factor  
        if random.random() < self.mutation_rate:  
            mutated_use_phase_factor = not use_phase_factor # Flip the boolean value

        return (mutated_disp_scale, mutated_ratio, mutated_cell_transform_vec, mutated_supercell_variant, mutated_use_phase_factor)

    def evolve(self, current_population_with_fitness):
        """Evolves the population for one generation."""
        self.population = current_population_with_fitness
        
        # Sort population by fitness (energy), lowest energy is best
        # Handle cases where fitness might be None
        valid_population = [p for p in self.population if p['fitness'] is not None]
        if not valid_population:
             print("No valid fitness values in population to evolve. Generating random offspring.")
             return [self._generate_random_individual() for _ in range(self.num_offspring)]

        valid_population.sort(key=lambda x: x['fitness'])
        self.population = valid_population # Update population to only include valid ones

        new_offspring_params = []
        
        # Elitism: Keep the single best individual
        new_offspring_params.append(self.population[0]['params'])

        # Generate the rest of the offspring
        while len(new_offspring_params) < self.num_offspring:
            try:
                parent1_params, parent2_params = self._select_parents()
            except ValueError as e:
                print(f"Error during parent selection: {e}. Generating a random individual instead.")
                new_offspring_params.append(self._generate_random_individual())
                continue

            child1_params, child2_params = self._crossover(parent1_params, parent2_params)

            new_offspring_params.append(self._mutate(child1_params))
            if len(new_offspring_params) < self.num_offspring:
                new_offspring_params.append(self._mutate(child2_params))
        
        print(f"Generated {len(new_offspring_params)} new offspring for the next generation.")
        return new_offspring_params