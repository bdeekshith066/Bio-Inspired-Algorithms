import random
import numpy as np
import matplotlib.pyplot as plt

class AntColonyOptimization:
    def __init__(self, cities, n_ants, n_iterations, alpha, beta, rho, q0):
        # Initialize the ACO class with the given parameters
        self.cities = cities  # Coordinates of cities (nodes in the graph)
        self.n_cities = len(cities)  # Number of cities
        self.n_ants = n_ants  # Number of ants
        self.n_iterations = n_iterations  # Number of iterations
        self.alpha = alpha  # Importance of pheromone in decision-making
        self.beta = beta  # Importance of heuristic (distance) in decision-making
        self.rho = rho  # Rate of pheromone evaporation
        self.q0 = q0  # Probability of exploitation (vs exploration)

        # Initialize pheromone levels as a matrix with ones
        self.pheromone = np.ones((self.n_cities, self.n_cities))

        # Compute the distance matrix between all cities
        self.distances = self.compute_distances()

    def compute_distances(self):
        # Calculate the distance between every pair of cities
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                # Use Euclidean distance formula
                distance = np.linalg.norm(self.cities[i] - self.cities[j])
                distances[i][j] = distances[j][i] = distance  # Symmetric distances
        return distances

    def select_next_city(self, current_city, visited):
        # Decide the next city for the ant to move to
        probabilities = np.zeros(self.n_cities)  # To store probabilities for each city
        tau = self.pheromone[current_city]  # Pheromone levels from the current city
        eta = 1.0 / (self.distances[current_city] + 1e-10)  # Heuristic info (inverse of distance)

        for i in range(self.n_cities):
            if i not in visited:  # Only consider cities not yet visited
                probabilities[i] = (tau[i] ** self.alpha) * (eta[i] ** self.beta)

        probabilities_sum = probabilities.sum()
        probabilities /= probabilities_sum  # Normalize probabilities to sum to 1

        if random.random() < self.q0:
            # Exploitation: Choose the city with the highest probability
            next_city = np.argmax(probabilities)
        else:
            # Exploration: Choose probabilistically based on probabilities
            next_city = np.random.choice(range(self.n_cities), p=probabilities)

        return next_city

    def construct_solution(self):
        # Build a tour for one ant
        visited = [random.randint(0, self.n_cities - 1)]  # Start at a random city
        tour = visited[:]  # Initialize the tour with the starting city

        while len(visited) < self.n_cities:
            current_city = visited[-1]  # Current position of the ant
            next_city = self.select_next_city(current_city, visited)  # Decide the next city
            visited.append(next_city)  # Mark it as visited
            tour.append(next_city)  # Add it to the tour

        tour.append(tour[0])  # Return to the starting city to complete the cycle
        return tour

    def update_pheromones(self, ants_solutions, ants_lengths):
        # Update the pheromone matrix based on the ants' tours
        self.pheromone *= (1 - self.rho)  # Evaporate pheromone (reduce all values by a factor of rho)

        for i in range(self.n_ants):
            solution = ants_solutions[i]  # The tour completed by an ant
            length = ants_lengths[i]  # Total distance of this tour

            for j in range(self.n_cities):
                from_city = solution[j]
                to_city = solution[j + 1]
                # Increase pheromone on the edges used by this ant, proportional to 1 / length
                self.pheromone[from_city][to_city] += 1.0 / length
                self.pheromone[to_city][from_city] += 1.0 / length  # Symmetric pheromone update

    def optimize(self):
        # The main optimization loop
        best_solution = None  # Store the best tour found
        best_length = float('inf')  # Initialize the best length as infinity
        all_lengths = []  # Keep track of the best length in each iteration

        for iteration in range(self.n_iterations):
            ants_solutions = []  # Store tours created by all ants in this iteration
            ants_lengths = []  # Store lengths of these tours

            for _ in range(self.n_ants):
                solution = self.construct_solution()  # Build a tour
                length = self.calculate_total_length(solution)  # Calculate its length
                ants_solutions.append(solution)  # Save the tour
                ants_lengths.append(length)  # Save the length

                if length < best_length:
                    # Update the best solution if this tour is shorter
                    best_solution = solution
                    best_length = length

            self.update_pheromones(ants_solutions, ants_lengths)  # Update pheromones based on tours

            all_lengths.append(best_length)  # Track the best length of this iteration
            print(f"Iteration {iteration + 1}, Best Length: {best_length}")

        return best_solution, best_length, all_lengths

    def calculate_total_length(self, solution):
        # Calculate the total distance of a tour
        total_length = 0
        for i in range(self.n_cities):
            from_city = solution[i]
            to_city = solution[i + 1]
            total_length += self.distances[from_city][to_city]  # Sum the distances between consecutive cities
        return total_length

    def plot_solution(self, solution):
        # Visualize the best solution (tour)
        x = [self.cities[city][0] for city in solution]  # X-coordinates of the cities in the tour
        y = [self.cities[city][1] for city in solution]  # Y-coordinates of the cities in the tour
        plt.plot(x, y, marker='o')  # Draw the tour as a line connecting the cities
        plt.plot([x[0], x[-1]], [y[0], y[-1]], marker='o', linestyle="--", color='r')  # Close the loop
        plt.title(f"Best Tour Length: {self.calculate_total_length(solution):.2f}")
        plt.show()

# Define the cities (as an example, you can change the coordinates)
cities = np.array([
    [0, 0],
    [1, 3],
    [3, 1],
    [5, 3],
    [6, 6],
    [8, 3],
    [9, 0],
    [7, -2]
])

# Parameters for the ACO algorithm
n_ants = 10  # Number of ants
n_iterations = 100  # Number of iterations
alpha = 1.0  # Influence of pheromone
beta = 2.0   # Influence of distance (heuristic)
rho = 0.5    # Pheromone evaporation rate
q0 = 0.9     # Probability of exploitation (vs exploration)

# Initialize and run the ACO algorithm
aco = AntColonyOptimization(cities, n_ants, n_iterations, alpha, beta, rho, q0)
best_solution, best_length, all_lengths = aco.optimize()

# Output the best solution
print(f"Best Solution: {best_solution}")
print(f"Best Length: {best_length}")

# Plot the best solution
aco.plot_solution(best_solution)
