import numpy as np
import heapq

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from keras.src.models import Sequential
from keras.src.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.src.layers import ReLU, Activation, ELU, LeakyReLU

from keras.src.callbacks import Callback

class TQDMProgressBar(Callback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)

'''
Chromosome bits are organized as follows:
[0-1] -> num convolutional layers
[2-9] -> conv layer 1 params
    [2-3] -> num filters
    [4] -> kernel size
    [5-6] -> activation function
    [7-8] -> pooling
    [9] -> batch normalization
[10-17] -> conv layer 2 params
    [10-11] -> num filters
    [12] -> kernel size
    [13-14] -> activation function
    [15-16] -> pooling
    [17] -> batch normalization
[18-25] -> conv layer 3 params
    [18-19] -> num filters
    [20] -> kernel size
    [21-22] -> activation function
    [23-24] -> pooling
    [25] -> batch normalization
[26-33] -> conv layer 4 params
    [26-27] -> num filters
    [28] -> kernel size
    [29-30] -> activation function
    [31-32] -> pooling
    [33] -> batch normalization
[34-35] -> num dense layers
[36-41] -> dense layer 1 params
    [36-37] -> num neurons
    [38-39] -> activation function
    [40-41] -> dropout percent
[42-47] -> dense layer 2 params
    [42-43] -> num neurons
    [44-45] -> activation function
    [46-47] -> dropout percent
[48-53] -> dense layer 3 params
    [48-49] -> num neurons
    [50-51] -> activation function
    [52-53] -> dropout percent
[54-59] -> dense layer 4 params
    [54-55] -> num neurons
    [56-57] -> activation function
    [58-59] -> dropout percent
'''
CHROMOSOME_LEN = 60

FILTERS = [16, 32, 64, 128]
KERNELS = [(3,3), (5,5)]
ACTIVATION_FUNCTIONS = ['relu', 'tanh', 'elu', 'leaky-relu']
POOLING = ['max', 'avg', None, None]
BATCH_NORM = [True, False]
NEURONS = [64, 128, 256, 512]
DROPOUT = [0.0, 0.2, 0.3, 0.5]

BALANCE_PCT = 0.8

BATCH_SIZE = 64

class QuantumPopulationManager:

    def __init__(self, input_shape, generations):
        self.input_shape = input_shape
        self.gens = generations
        
        # Keep consistent random state per instance
        self.random_state = np.random.randint(0, 0x7FFFFFFF)

    def init_population(self):
        '''
        Generates a population of chromosomes having equal alpha/beta values.
        '''
        self.population = np.ones((self.pop_size, CHROMOSOME_LEN, 2)) / np.sqrt(2)
    
    def measure_chromosome(self, chromosome):
        '''
        Helper function to measure the values of the quantum bits

        Inputs: Quantum chromosome

        Returns: Measured chromosome bits
        '''

        measured_chromosome = np.zeros(CHROMOSOME_LEN, dtype=int)

        for i in range(CHROMOSOME_LEN):
            if np.random.rand() < chromosome[i][0]**2:
                measured_chromosome[i] = 0
            else:
                measured_chromosome[i] = 1
        return measured_chromosome


    def get_model_architecture_from_chromosome(self, chromosome):
        '''
        Generates the model structure from the bits of the chromosome

        Inputs: Chromosome whose bits to use

        Returns: The corresponding model structure as a dictionary
        '''
        architecture = {}
        chrome_index = 0

        # Get the number of convolutional layers
        num_convolutional = ((chromosome[chrome_index] << 1) + chromosome[chrome_index+1]) + 1
        architecture['num_convolutional_layers'] = num_convolutional
        chrome_index += 2

        # Get parameters for each convolutional layer
        for i in range(num_convolutional):
            filter_index = (chromosome[chrome_index] << 1) + chromosome[chrome_index+1]
            kernel_index = chromosome[chrome_index+2]
            activation_index = (chromosome[chrome_index+3] << 1) + chromosome[chrome_index+4]
            pooling_index = (chromosome[chrome_index+5] << 1) + chromosome[chrome_index+6]
            batch_norm_index = chromosome[chrome_index+7]

            chrome_index += 8

            architecture[f'layer_{i+1}_num_filters']       = FILTERS[filter_index]
            architecture[f'layer_{i+1}_kernel']            = KERNELS[kernel_index]
            architecture[f'layer_{i+1}_conv_activation']   = ACTIVATION_FUNCTIONS[activation_index]
            architecture[f'layer_{i+1}_pooling']           = POOLING[pooling_index]
            architecture[f'layer_{i+1}_batch_norm']        = BATCH_NORM[batch_norm_index]
        
        chrome_index = 34

        # Get the number of dense layers
        num_dense = ((chromosome[chrome_index] << 1) + chromosome[chrome_index+1]) + 1
        architecture['num_dense_layers'] = num_dense
        chrome_index += 2

        # Get parameters for each dense layer
        for i in range(num_dense):
            neuron_index = (chromosome[chrome_index] << 1) + chromosome[chrome_index+1]
            activation_index = (chromosome[chrome_index+2] << 1) + chromosome[chrome_index+3]
            dropout_index = (chromosome[chrome_index+4] << 1) + chromosome[chrome_index+5]

            chrome_index += 6

            architecture[f'layer_{i+1}_num_neurons']        = NEURONS[neuron_index]
            architecture[f'layer_{i+1}_dense_activation']   = ACTIVATION_FUNCTIONS[activation_index]
            architecture[f'layer_{i+1}_dropout']            = DROPOUT[dropout_index]

        return architecture

    def build_model_from_architecture(self, architecture):
        '''
        Builds the model from the given architecture

        Inputs: Architecture dictionary outlining model structure

        Returns: Corresponding model
        '''

        # Start model
        model = Sequential()
        model.add(Input(shape=self.input_shape))

        # Add convolutional layers
        num_convolutional = architecture['num_convolutional_layers']
        for i in range(num_convolutional):
            filters = architecture[f'layer_{i+1}_num_filters']
            kernel = architecture[f'layer_{i+1}_kernel']
            model.add(Conv2D(filters, kernel, padding='same'))

            activation_func = architecture[f'layer_{i+1}_conv_activation']
            if activation_func == 'relu':
                model.add(ReLU())
            elif activation_func == 'tanh':
                model.add(Activation('tanh'))
            elif activation_func == 'elu':
                model.add(ELU(alpha=1.0))
            elif activation_func == 'leaky-relu':
                model.add(LeakyReLU(negative_slope=0.01))
            
            pooling = architecture[f'layer_{i+1}_pooling']
            if pooling == 'max':
                model.add(MaxPooling2D(pool_size=(2,2)))
            elif pooling == 'avg':
                model.add(AveragePooling2D(pool_size=(2,2)))

            batch_norm = architecture[f'layer_{i+1}_batch_norm']
            if batch_norm:
                model.add(BatchNormalization())

        # Flatten 2D image
        model.add(Flatten())

        # Add dense layers
        num_dense = architecture['num_dense_layers']
        for i in range(num_dense):
            neurons = architecture[f'layer_{i+1}_num_neurons']
            model.add(Dense(neurons))

            activation_func = architecture[f'layer_{i+1}_dense_activation']
            if activation_func == 'relu':
                model.add(ReLU())
            elif activation_func == 'tanh':
                model.add(Activation('tanh'))
            elif activation_func == 'elu':
                model.add(ELU(alpha=1.0))
            elif activation_func == 'leaky-relu':
                model.add(LeakyReLU(negative_slope=0.01))
            
            dropout = architecture[f'layer_{i+1}_dropout']
            if dropout != 0.0:
                model.add(Dropout(dropout))

        # Add output layer
        model.add(Dense(10, activation='softmax'))

        return model

    def get_fitness(self, acc, complexity):
        '''
        Helper function to compute the fitness of a model.
        This depends on the accuracy on the validation set and the number
        of trained parameters.

        Inputs:  Model validation accuracy
                 Model complexity (# of params)

        Returns: Model fitness score
        '''

        MIN_COMPLEXITY, MAX_COMPLEXITY = 5e4, 5e7  # From 50K to 50M params

        scaled_acc = acc ** 0.5  # Take the root of accuracy to incentivize early growth
        scaled_complexity = (np.log10(complexity + 1) - np.log10(MIN_COMPLEXITY)) / (np.log10(MAX_COMPLEXITY) - np.log10(MIN_COMPLEXITY))

        raw_score = BALANCE_PCT * scaled_acc - (1-BALANCE_PCT) * scaled_complexity
        true_score = raw_score / BALANCE_PCT  # This makes the values fit to range [-1+1/w, 1]
        return true_score

    def evaluate_model(self, model: Sequential, data, pbar):
        '''
        Trains and evaluates the model.

        Inputs: The model to train
                The dataset (X_train, y_train, X_val, y_val)

        Returns: The fitness value and classification accuracy.
        '''
        X_train, y_train, X_val, y_val = data

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=BATCH_SIZE,
                            validation_data=(X_val, y_val), verbose=0,
                            callbacks=[TQDMProgressBar(pbar)])
        
        acc = history.history['val_accuracy'][-1]
        complexity = model.count_params()

        fitness = self.get_fitness(acc, complexity)
        return fitness, acc

    def get_weighted_target(self, top_chromosomes):
        '''
        Gets the target chromosome as a weighted sum of the top K

        Inputs:  The top K chromosomes from the population

        Returns: The weighted top chromosome
        '''
        weighted_chromosome = np.sum(self.top_weights.reshape((self.k, 1)) * top_chromosomes, axis=0)
        return weighted_chromosome


    def quantum_rotation_gate(self, population, classical_pop, best_chromosome):
        '''
        Updates the population to reflect the best model

        Inputs:  The quantum population
                 The classical measured popluation
                 The best chromosome architecture found

        Returns: The new updated population
        '''
        for i, chromosome in enumerate(population):
            for j, qubit in enumerate(chromosome):
                observed_bit = classical_pop[i][j]
                best_bit = best_chromosome[j]

                # Update qubit if it does not match target
                if (observed_bit == 0 and best_bit > 0.5) or (observed_bit == 1 and best_bit < 0.5):
                    # Weigh the delta theta value based on the distance to the target
                    delta_theta = self.theta * (best_bit - 0.5) * 2

                    alpha, beta = qubit
                    new_alpha = alpha * np.cos(delta_theta) - beta * np.sin(delta_theta)
                    new_beta = alpha * np.sin(delta_theta) + beta * np.cos(delta_theta)
                    population[i][j] = [new_alpha, new_beta]
        return population
    

    def update_phase(self, generation, data):
        '''
        Checks the current generation to decide if we are in the Exploration, Focused, or Convergence phase.
        Increases dataset size and epochs, decreases theta and population size
        '''
        # Update parameters based on generation
        if generation < 5:
            self.dataset_size = 0.2
            self.pop_size = 25
            self.k = 4
            self.epochs = 5
            self.theta = 0.05
        elif generation < 10:
            self.dataset_size = 0.5
            self.pop_size = 12
            self.k = 3
            self.epochs = 12
            self.theta = 0.025
        else: # generation < 15
            self.dataset_size = 1
            self.pop_size = 5
            self.k = 2
            self.epochs = 30
            self.theta = 0.01

        # Update weights to match new K
        raw_weights = 1 / (1 << np.arange(self.k))
        self.top_weights = raw_weights / raw_weights.sum()

        (X_train, y_train, X_test, y_test) = data

        if self.dataset_size == 1:
            data_slice = data
        else:
            X_train_small, _, y_train_small, _ = train_test_split(
                X_train, y_train, 
                train_size=int(y_train.shape[0]*self.dataset_size), 
                stratify=y_train,
                random_state=self.random_state
            )

            X_test_small, _, y_test_small, _ = train_test_split(
                X_test, y_test, 
                train_size=int(y_test.shape[0]*self.dataset_size), 
                stratify=y_test,
                random_state=self.random_state
            )

            data_slice = (X_train_small, y_train_small, X_test_small, y_test_small)

        return data_slice


    def run_evolution(self, data):
        '''
        Runs the evolution process on the population.

        Inputs: The dataset (X_train, y_train, X_val, y_val)

        Returns: The best model architecture and best fitness value found
        '''

        # Holds top k fitnesses, accuracies, chromosomes, architectures
        top_k = []

        # Hold top fitness, mean fitness, weighted top fitness, top accuracy, mean accuracy for each generation
        generational_scores = []

        # Get initial hyperparmeters and data
        data_slice = self.update_phase(0, data)

        # Initialize population
        self.init_population()

        for gen in range(self.gens):

            classical_pop = []

            with tqdm(total=self.epochs*self.pop_size, desc="Training QGA Models") as pbar:

                for chromosome in self.population:
                    classical_chromosome = self.measure_chromosome(chromosome)
                    architecture = self.get_model_architecture_from_chromosome(classical_chromosome)
                    model = self.build_model_from_architecture(architecture)
                    fitness, accuracy = self.evaluate_model(model, data_slice, pbar)

                    pbar.set_postfix(last_fitness=f"{fitness:.4f}", last_acc=f"{accuracy:.4f}", last_params=f"{model.count_params()}")

                    classical_pop.append(classical_chromosome)

                    heapq.heappush(top_k, (fitness, accuracy, classical_chromosome, architecture))

            # Hold generational data
            fitness_scores = [t[0] for t in top_k]
            accuracies = [t[1] for t in top_k]

            # Get top k for current generation
            top_k = heapq.nlargest(self.k, top_k)
            top_chromosomes = [t[2] for t in top_k]

            # Record generational data
            top_fitness = top_k[0][0]
            mean_fitness = np.mean(fitness_scores)
            weighted_top_fitness = np.sum(self.top_weights * np.array([t[0] for t in top_k]))
            top_acc = top_k[0][1]
            mean_acc = np.mean(accuracies)
            weighted_top_acc = np.sum(self.top_weights * np.array([t[1] for t in top_k]))
            generational_scores.append((top_fitness, mean_fitness, weighted_top_fitness, top_acc, mean_acc, weighted_top_acc))

            print(f"Generation {gen+1}: Top K = {[(t[0],t[1]) for t in top_k]}")
            print(f"Generation {gen+1}: Top Fitness = {generational_scores[-1][0]}, Top Accuracy = {generational_scores[-1][3]}")
            print(f"Generation {gen+1}: Mean Fitness = {generational_scores[-1][1]}, Mean Accuracy = {generational_scores[-1][4]}")

            top_model = self.build_model_from_architecture(top_k[0][3])
            print(f"Generation {gen+1}: Top Model Complexity = {top_model.count_params()}")
            print()

            # Weigh top-k chromosomes to get new target
            weighted_chromosome = self.get_weighted_target(top_chromosomes)

            # Update hyperparams for next update
            data_slice = self.update_phase(gen+1, data)

            # Need to cull population
            if self.pop_size != len(self.population):
                # Sort population by fitness
                top_pop = sorted(zip(fitness_scores, classical_pop, self.population), key=lambda x: x[0], reverse=True)
                classical_pop = [chromosome for _, chromosome, _ in top_pop[:self.pop_size]]
                self.population = [chromosome for _, _, chromosome in top_pop[:self.pop_size]]

            # Update population
            self.population = self.quantum_rotation_gate(self.population, classical_pop, weighted_chromosome)

        best_fitness, best_acc, best_chromosome, best_architecture = top_k[0]
        return best_architecture, best_fitness, generational_scores
