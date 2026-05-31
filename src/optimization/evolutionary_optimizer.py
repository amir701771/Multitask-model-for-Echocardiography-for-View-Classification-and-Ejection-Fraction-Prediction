
import random
import logging
import os
import json
import numpy as np
import time

try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("DEAP library not found. Please install it: pip install deap")
    raise

from src.optimization.objective_multitask import objective_function_multitask

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class EvolutionaryOptimizer:

    def __init__(self, base_config, ea_config_data):

        self.base_config = base_config

        if 'ea' not in ea_config_data:
             raise ValueError("EA configuration YAML must contain a top-level 'ea' key.")
        self.ea_config = ea_config_data['ea']
        self.param_space = self.ea_config['param_space']

        multitask_output_dir = self.base_config['training']['output_dir']
        self.output_dir = os.path.join(multitask_output_dir, "ea_optimization")
        os.makedirs(self.output_dir, exist_ok=True)

        logging.info("--- Initializing Evolutionary Optimizer (DEAP) for MultiTask Model ---")
        logging.info(f"Base MultiTask Config Output Dir: {multitask_output_dir}")
        logging.info(f"EA Results Dir: {self.output_dir}")
        logging.info(f"Parameters to Optimize: {list(self.param_space.keys())}")

        self._setup_deap()

        logging.info("DEAP setup complete.")
        pop = self.ea_config['population_size']; gen = self.ea_config['generations']
        cxpb = self.ea_config['cxpb']; mutpb = self.ea_config['mutpb']
        logging.info(f"EA settings: Pop={pop}, Gen={gen}, CXPB={cxpb}, MUTPB={mutpb}")


    def _setup_deap(self):

        if hasattr(creator, "FitnessMin"): del creator.FitnessMin
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # -1.0 for minimization

        if hasattr(creator, "Individual"): del creator.Individual
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        self.param_names = []
        self.param_bounds = {'low': [], 'up': []}
        self.param_types = {}

        logging.info("Registering parameters in DEAP toolbox:")
        for name, settings in self.param_space.items():
            self.param_names.append(name)
            ptype = settings['type'].lower()
            self.param_types[name] = ptype
            logging.info(f"  - {name} (type: {ptype}, settings: {settings})")

            low, high = settings.get('low'), settings.get('high') # Use .get for safe access
            options = settings.get('options')

            if ptype == "float":
                if low is None or high is None: raise ValueError(f"Float param '{name}' needs 'low' and 'high'.")
                self.toolbox.register(f"attr_{name}", random.uniform, low, high)
                self.param_bounds['low'].append(float(low))
                self.param_bounds['up'].append(float(high))
            elif ptype == "int":
                if low is None or high is None: raise ValueError(f"Int param '{name}' needs 'low' and 'high'.")
                self.toolbox.register(f"attr_{name}", random.randint, int(low), int(high))
                self.param_bounds['low'].append(int(low))
                self.param_bounds['up'].append(int(high))
            elif ptype == "choice":
                if options is None or not isinstance(options, list): raise ValueError(f"Choice param '{name}' needs a list of 'options'.")
                self.toolbox.register(f"attr_{name}", random.choice, options)

                self.param_bounds['low'].append(None)
                self.param_bounds['up'].append(None)
            else:
                raise ValueError(f"Unsupported parameter type '{ptype}' for parameter '{name}' in ea_config.yaml")

        logging.info(f"Parameter registration order: {self.param_names}")

        attr_gens = [getattr(self.toolbox, f"attr_{name}") for name in self.param_names]
        self.toolbox.register("individual", tools.initCycle, creator.Individual, attr_gens)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self._evaluate_individual)

        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                              low=self.param_bounds['low'],
                              up=self.param_bounds['up'],
                              eta=20.0)

        mutate_indpb = 1.0 / len(self.param_names) if self.param_names else 0.1
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                              low=self.param_bounds['low'],
                              up=self.param_bounds['up'],
                              eta=20.0,
                              indpb=mutate_indpb)

        self.toolbox.register("select", tools.selTournament, tournsize=3)


    def _evaluate_individual(self, individual):

        hyperparameters = {name: value for name, value in zip(self.param_names, individual)}

        run_id = f"gen{self.current_gen}_eval{self.eval_counter}_{int(time.time()*1000)}"
        self.eval_counter += 1 # Increment evaluation counter for the current generation

        repaired = False
        for i, name in enumerate(self.param_names):
             ptype = self.param_types[name]
             low, up = self.param_bounds['low'][i], self.param_bounds['up'][i]

             if ptype in ["float", "int"] and low is not None and up is not None:
                 current_val = individual[i]

                 clamped_val = max(low, min(current_val, up))

                 if ptype == "int":
                     clamped_val = int(round(clamped_val))

                 if clamped_val != individual[i]:
                      individual[i] = clamped_val
                      hyperparameters[name] = clamped_val
                      repaired = True


        if repaired:
             logging.debug(f"EA Run {run_id}: Repaired individual post-operator. New params: {hyperparameters}")

        fitness_tuple = objective_function_multitask(hyperparameters, self.base_config, run_id=run_id)

        return fitness_tuple


    def run(self):

        pop_size = self.ea_config['population_size']
        num_generations = self.ea_config['generations']
        cxpb = self.ea_config['cxpb'] # Crossover probability
        mutpb = self.ea_config['mutpb'] # Mutation probability

        logging.info(f"--- Starting Evolutionary Optimization Run ---")
        logging.info(f"Population Size: {pop_size}, Generations: {num_generations}")
        logging.info(f"Crossover P(CXPB): {cxpb}, Mutation P(MUTPB): {mutpb}")

        population = self.toolbox.population(n=pop_size)

        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        logging.info("Evaluating initial population (Generation 0)...")
        self.current_gen = 0
        self.eval_counter = 0
        invalid_ind = [ind for ind in population if not ind.fitness.valid]

        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)

        logging.info(f"GEN 0 | Evals: {len(invalid_ind)} | {logbook.stream}")

        for gen in range(1, num_generations + 1):
            self.current_gen = gen
            self.eval_counter = 0
            start_gen_time = time.time()
            logging.info(f"--- Starting Generation {gen}/{num_generations} ---")

            offspring = self.toolbox.select(population, len(population))

            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    self.toolbox.mate(child1, child2)

                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    self.toolbox.mutate(mutant)

                    del mutant.fitness.values


            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                 num_evals = len(invalid_ind)
                 logging.info(f"Evaluating {num_evals} new individuals for generation {gen}...")
                 fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                 for ind, fit in zip(invalid_ind, fitnesses):
                     ind.fitness.values = fit
                 logging.info(f"Finished evaluating new individuals for generation {gen}.")
            else:
                 num_evals = 0
                 logging.info(f"No new individuals to evaluate for generation {gen}.")


            population[:] = offspring

            hof.update(population)
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=num_evals, **record)

            gen_time = time.time() - start_gen_time

            best_overall_fitness = hof[0].fitness.values[0] if hof else float('nan')
            logging.info(f"GEN {gen} | Evals: {num_evals} | {logbook.stream} | Best Overall: {best_overall_fitness:.6f} | Time: {gen_time:.2f}s")

            if gen % 5 == 0 or gen == num_generations:
                self._save_results(hof, logbook, f"ea_results_gen_{gen}.json")


        logging.info("--- Evolutionary Algorithm Finished ---")

        if hof:
            best_individual = hof[0]
            best_fitness = best_individual.fitness.values[0]

            best_params = {name: value for name, value in zip(self.param_names, best_individual)}

            logging.info("="*30 + " BEST RESULT " + "="*30)
            logging.info(f"Best Individual Fitness (Validation Loss): {best_fitness:.6f}")
            logging.info(f"Best Hyperparameters Found:")
            logging.info(f"{json.dumps(best_params, indent=4)}")
            logging.info("="*73)


            self._save_results(hof, logbook, "ea_results_final.json")

            return best_params, best_fitness
        else:
             logging.warning("Hall of Fame is empty at the end of the run.")
             logging.warning("This might happen if all objective function evaluations failed.")
             return None, float('inf') # Return indication of failure


    def _save_results(self, hall_of_fame, logbook, filename):
        """Saves the best parameters and logbook history to a JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        results_data = {
            'best_individual': None,

            'logbook': logbook.chapters if hasattr(logbook, 'chapters') else logbook
        }
        if hall_of_fame:
             best_ind_list = hall_of_fame[0]
             results_data['best_individual'] = {

                 'parameters': {name: value for name, value in zip(self.param_names, best_ind_list)},
                 'fitness': best_ind_list.fitness.values[0] # Assuming single objective fitness
             }

        try:
            with open(filepath, 'w') as f:

                json.dump(results_data, f, indent=4, cls=NumpyEncoder)
            logging.info(f"EA results snapshot saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving EA results to {filepath}: {e}")