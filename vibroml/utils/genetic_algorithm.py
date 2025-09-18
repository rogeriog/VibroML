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
                 tournament_size=10,
                 tracked_k_points_data=None
                ):
        """
        Initializes the Genetic Algorithm.

        Args:
            tracked_k_points_data (dict, optional): Comprehensive tracking data for mode replacement:
                {
                  'soft_modes': [list of all soft modes below threshold],
                  'highest_freq_modes': [list of highest frequency modes at special k-points],
                  'lowest_freq_modes': [list of lowest frequency modes at special k-points, excluding Gamma]
                }
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

        # Mode tracking for second mode replacement
        self.tracked_k_points_data = tracked_k_points_data or {
            'soft_modes': [],
            'highest_freq_modes': [],
            'lowest_freq_modes': []
        }

        # The population will be a list of dictionaries:
        # [{'params': (disp_scale, ratio, cell_transform_vec, supercell_variant, use_phase_factor),
        #   'fitness': energy, 'mutation_data': {...}}, ...]
        self.population = []

    def _generate_random_individual(self):
        """Generates a single random individual (parameter set) within bounds.

        Returns:
            tuple: (individual_params, mutation_data) where:
                - individual_params: (disp_scale, ratio, cell_transform_vec, supercell_variant, use_phase_factor)
                - mutation_data: dict containing mutation tracking information
        """
        disp_scale = random.uniform(*self.displacement_scale_bounds)
        ratio = random.uniform(*self.ratio_mode2_to_mode1_bounds)
        cell_scales = [random.uniform(*self.cell_scale_bounds) for _ in range(3)]
        cell_angles = [random.uniform(*self.cell_angle_bounds) for _ in range(3)]
        cell_transformation_vector = tuple(cell_scales + cell_angles)

        # Pick a random supercell variant from the provided list
        supercell_variant = random.choice(self.supercell_variants)
        use_phase_factor = random.choices([True, False], weights=[1-self.phase_factor_off_ratio, self.phase_factor_off_ratio])[0]

        # Probabilistic second mode replacement during initialization
        mode_replaced = False
        selected_mode = None

        if random.random() < self.mutation_rate:
            selected_mode = self._select_random_mode_for_replacement()
            if selected_mode:
                mode_replaced = True

        mutation_data = self._create_mutation_data(mode_replaced, selected_mode)
        individual_params = (disp_scale, ratio, cell_transformation_vector, supercell_variant, use_phase_factor)

        return individual_params, mutation_data

    def _get_available_modes_for_replacement(self):
        """Get all available modes for second mode replacement."""
        available_modes = []

        # Add soft modes if available
        if self.tracked_k_points_data['soft_modes']:
            available_modes.extend(self.tracked_k_points_data['soft_modes'])

        # If no soft modes, add frequency extrema modes
        if not available_modes:
            # Add highest frequency modes
            if self.tracked_k_points_data['highest_freq_modes']:
                available_modes.extend(self.tracked_k_points_data['highest_freq_modes'])

            # Add lowest frequency modes (excluding Gamma)
            if self.tracked_k_points_data['lowest_freq_modes']:
                available_modes.extend(self.tracked_k_points_data['lowest_freq_modes'])

        return available_modes

    def _select_random_mode_for_replacement(self):
        """Randomly select a mode from available modes for replacement."""
        available_modes = self._get_available_modes_for_replacement()

        if not available_modes:
            return None

        selected_mode = random.choice(available_modes)
        return {
            'label': selected_mode['label'],
            'coordinate': selected_mode['coordinate'],
            'frequency': selected_mode['frequency'],
            'band_index': selected_mode['band_index'],
            'kpoint_index_in_path': selected_mode['kpoint_index_in_path']
        }

    def _create_mutation_data(self, mode_replaced=False, selected_mode=None):
        """Create mutation tracking data for an individual."""
        return {
            'mode_replaced': mode_replaced,
            'selected_mode': selected_mode.copy() if selected_mode else None
        }

    def initialize_population(self, initial_individuals=None):
        """
        Initializes the population.
        If initial_individuals are provided, they form the starting population.
        Otherwise, a random population is generated.
        """
        self.population = []
        if initial_individuals:
            # Ensure all initial individuals have mutation_data for backward compatibility
            for ind in initial_individuals:
                if 'mutation_data' not in ind:
                    # Add default mutation_data for backward compatibility
                    ind['mutation_data'] = {'mode_replaced': False, 'selected_mode': None}

            self.population.extend(initial_individuals)
            # If initial_individuals are fewer than population_size, fill the rest randomly
            while len(self.population) < self.population_size:
                individual_params, mutation_data = self._generate_random_individual()
                self.population.append({
                    'params': individual_params,
                    'fitness': None,
                    'mutation_data': mutation_data
                })
        else:
            for _ in range(self.population_size):
                individual_params, mutation_data = self._generate_random_individual()
                self.population.append({
                    'params': individual_params,
                    'fitness': None,
                    'mutation_data': mutation_data
                })

        # Count and report mode replacements during initialization
        mode_replacements = sum(1 for ind in self.population if ind.get('mutation_data', {}).get('mode_replaced', False))
        print(f"Initialized GA population with {len(self.population)} individuals.")
        print(f"Mode replacements during initialization: {mode_replacements}/{len(self.population)}")

    def get_population_with_mutation_data(self):
        """Returns the current population parameters with their corresponding mutation data.

        Returns:
            tuple: (list of parameters, list of mutation data)
        """
        params_list = [ind['params'] for ind in self.population]
        mutation_data_list = [ind['mutation_data'] for ind in self.population]
        return params_list, mutation_data_list

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
        """Mutates numerical and categorical genes of an individual.

        Returns:
            tuple: (mutated_individual_params, mutation_data) where:
                - mutated_individual_params: (disp_scale, ratio, cell_transform_vec, supercell_variant, use_phase_factor)
                - mutation_data: dict containing mutation tracking information
        """
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

        # 4. Probabilistic second mode replacement during mutation
        mode_replaced = False
        selected_mode = None

        if random.random() < self.mutation_rate:
            selected_mode = self._select_random_mode_for_replacement()
            if selected_mode:
                mode_replaced = True

        mutation_data = self._create_mutation_data(mode_replaced, selected_mode)
        mutated_individual_params = (mutated_disp_scale, mutated_ratio, mutated_cell_transform_vec, mutated_supercell_variant, mutated_use_phase_factor)

        return mutated_individual_params, mutation_data

    def evolve(self, current_population_with_fitness):
        """Evolves the population for one generation.

        Returns:
            tuple: (list of new offspring parameters, list of corresponding mutation data)
        """
        self.population = current_population_with_fitness

        # Sort population by fitness (energy), lowest energy is best
        # Handle cases where fitness might be None
        valid_population = [p for p in self.population if p['fitness'] is not None]
        if not valid_population:
             print("No valid fitness values in population to evolve. Generating random offspring.")
             offspring_with_data = [self._generate_random_individual() for _ in range(self.num_offspring)]
             return [params for params, _ in offspring_with_data]

        valid_population.sort(key=lambda x: x['fitness'])
        self.population = valid_population # Update population to only include valid ones

        new_offspring_params = []
        mutation_data_list = []

        # Elitism: Keep the single best individual (no mutation data for elite)
        new_offspring_params.append(self.population[0]['params'])
        mutation_data_list.append({'mode_replaced': False, 'selected_mode': None})

        # Generate the rest of the offspring
        while len(new_offspring_params) < self.num_offspring:
            try:
                parent1_params, parent2_params = self._select_parents()
            except ValueError as e:
                print(f"Error during parent selection: {e}. Generating a random individual instead.")
                individual_params, mutation_data = self._generate_random_individual()
                new_offspring_params.append(individual_params)
                mutation_data_list.append(mutation_data)
                continue

            child1_params, child2_params = self._crossover(parent1_params, parent2_params)

            # Mutate child1
            mutated_child1_params, child1_mutation_data = self._mutate(child1_params)
            new_offspring_params.append(mutated_child1_params)
            mutation_data_list.append(child1_mutation_data)

            # Mutate child2 if we still need more offspring
            if len(new_offspring_params) < self.num_offspring:
                mutated_child2_params, child2_mutation_data = self._mutate(child2_params)
                new_offspring_params.append(mutated_child2_params)
                mutation_data_list.append(child2_mutation_data)

        # Count and report mode replacements during evolution
        mode_replacements = sum(1 for data in mutation_data_list if data['mode_replaced'])
        print(f"Generated {len(new_offspring_params)} new offspring for the next generation.")
        print(f"Mode replacements during evolution: {mode_replacements}/{len(new_offspring_params)}")

        # Detailed logging for mode replacements
        if mode_replacements > 0:
            print("  Detailed mode replacement information:")
            for i, data in enumerate(mutation_data_list):
                if data['mode_replaced'] and data['selected_mode']:
                    selected_mode = data['selected_mode']
                    print(f"    Individual {i+1}: Second mode replaced with {selected_mode['label']} point, "
                          f"band {selected_mode['band_index']}, frequency {selected_mode['frequency']:.3f} THz, "
                          f"k-point index {selected_mode.get('kpoint_index_in_path', 'N/A')}")

        # Store mutation data for potential use in population summaries
        self.last_generation_mutation_data = mutation_data_list

        return new_offspring_params, mutation_data_list

    def get_mutation_summary(self):
        """Get summary of mutation data for the last generation.

        Returns:
            dict: Summary containing mutation statistics and details
        """
        if not hasattr(self, 'last_generation_mutation_data'):
            return {
                'total_individuals': 0,
                'mode_replacements': 0,
                'replacement_rate': 0.0,
                'selected_modes': []
            }

        mutation_data = self.last_generation_mutation_data
        total_individuals = len(mutation_data)
        mode_replacements = sum(1 for data in mutation_data if data['mode_replaced'])
        replacement_rate = mode_replacements / total_individuals if total_individuals > 0 else 0.0

        selected_modes = []
        for data in mutation_data:
            if data['mode_replaced'] and data['selected_mode']:
                selected_modes.append({
                    'label': data['selected_mode']['label'],
                    'frequency': data['selected_mode']['frequency'],
                    'band_index': data['selected_mode']['band_index'],
                    'kpoint_index_in_path': data['selected_mode'].get('kpoint_index_in_path', 'N/A'),
                    'coordinate': data['selected_mode'].get('coordinate', 'N/A')
                })

        return {
            'total_individuals': total_individuals,
            'mode_replacements': mode_replacements,
            'replacement_rate': replacement_rate,
            'selected_modes': selected_modes
        }