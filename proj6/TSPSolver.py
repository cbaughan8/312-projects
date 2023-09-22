#!/usr/bin/python3
from collections import defaultdict

from ReducedCostMatrix import ReducedCostMatrix
from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    pass
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

from TSPClasses import *
import heapq
import math


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    # Time Complexity: O(n^2), the reason for this is that we have to iterate through all the cities
    # and then iterate through all the cities again to find the nearest city from the current city
    # of the current iteration. All the logic  done within the inner loop is constant time, because
    # we are just comparing the distance of the current city to the nearest city.

    # Space Complexity: O(n^2), the reason for this is that when we retrieve the list of given cities it
    # takes up O(n) space, but while retrieving the matrix of edges it comes back as a 2D array which takes up
    # O(n^2) space. And finally when we build our best search so far tree it takes up O(n) because it has to store
    # the list of cities from out best search so far. The final result would be O(n^2) + O(n) + O(n) = O(n^2)
    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()  # Space Complexity: O(n) because we are storing all the cities in a list
        paths = self._scenario._edge_exists  # Space Complexity: O(n^2) because we are storing all the paths in a 2D array
        num_cities = len(cities)
        current_city = cities[0]

        solved_path = [current_city]  # Start with the first city
        total_cost = 0

        start_time = time.time()

        for i in range(num_cities):
            current_paths = paths[
                cities.index(current_city)]  # Time Complexity: O(1) because we are accessing the 2D array
            shortest_path = float('inf')  # Time Complexity: O(1)
            closest_city = None  # Time Complexity: O(1)

            for j in range(num_cities):  # Time Complexity: O(n)
                if current_paths[j]:
                    if cities[j] not in solved_path:
                        distance = current_city.costTo(cities[j])  # Time Complexity: O(1)
                        if distance < shortest_path:
                            shortest_path = distance
                            closest_city = cities[j]  # Time Complexity: O(1)

            if closest_city is not None:
                solved_path.append(closest_city)  # Time Complexity: O(1)
                current_city = closest_city
                total_cost += shortest_path

        bssf = TSPSolution(solved_path)  # Time Complexity: O(n), because it adds up the cost of the route
        end_time = time.time()

        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = 1
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

    # Time Complexity: O(n^2). Let me explain. To run greedy: O(n^2), to create matrix: O(n^2), To reduce O(n),
    # To push into the heap O(nlogn), The while loop O(n), To pop the queue O(log n), Add children to the
    # queue O(n logn)
    # While the simplified time complexity is O(n^2) there is a huge overhang on the time complexity. The unreduced
    # complexity is O(2n^2 + 2n + 2nlogn + logn)

    # Space Complexity: O(n^2). Let me explain. To run greedy: O(n^2), to create matrix: O(n^2), To reduce O(n^2),
    # To push into the heap O(n), The while loop O(n), To pop the queue O(n), Add children to the queue O(n).
    # The unreduced space complexity is O(3n^2 + 4n), reduced is O(n^2).

    def branchAndBound(self, time_allowance=60.0):
        # Initial setup of variables

        # We get our initial bssf by running the greedy algorithm
        # (Time Complexity: O(n^2))
        # (Space Complexity: O(n^2))
        bssf = self.greedy(time_allowance)

        number_of_solutions = 0
        pruned_states = 0

        max_queue_length = 1
        total_states = 1

        # We create the cost matrix for the given states
        # (Time Complexity: O(n^2))
        # (Space Complexity: O(n^2))
        matrix = self.create_matrix()

        # Time Complexity: O(n), let me explain, when we are initializing a reduced matrix we have to
        # iterate through all the rows and columns of the matrix by using two separate for loops. The
        # final result would be O(n) + O(n) = O(n)

        # Space Complexity: O(n^2), the reason for this is that we are creating a new matrix when
        # initializing the reduced matrix. The final result would be O(n^2) because of the 2D array
        # being created
        reduced_matrix = ReducedCostMatrix(matrix, [0])

        queue = []

        # Push reduced matrix elements into the empty queue with a heap push command
        # Time Complexity: O(NlogN) where N is the number of elements in the list
        # Space Complexity: O(N)
        heapq.heappush(queue, reduced_matrix)

        num_of_cities = len(self._scenario.getCities())
        cities = self._scenario.getCities()

        start_time = time.time()
        # We loop until the queue is empty, in other words, there are no more cities to visit or until the
        # designated time allowance is exceeded.
        # Time Complexity: O(n)
        # Space Complexity: O(n), space needed for varible cur_branch
        while len(queue) > 0 and time.time() - start_time < time_allowance:
            if len(queue) > max_queue_length:
                max_queue_length = len(queue)

            # Time Complexity: O(log n) where N is the number of elements in the list
            # Space Complexity: O(N)
            cur_branch = heapq.heappop(queue)

            # prune if more expensive than bssf
            if cur_branch.lower_bound >= bssf['cost']:
                pruned_states += 1
            # if path looped back to starting city
            elif cur_branch.visited.count(0) == 2 and cur_branch.num_visited() < num_of_cities:
                pruned_states += 1
            # check if a path exists to starting city
            elif not cur_branch.path_to_home():
                pruned_states += 1
            # possible bssf found
            elif cur_branch.num_visited() == num_of_cities:
                # if current solution is better than bssf
                if cur_branch.lower_bound < bssf['cost']:
                    # increment solution number
                    number_of_solutions += 1
                    # update bssf cost
                    bssf['cost'] = cur_branch.lower_bound
                    route = []
                    for i in cur_branch.visited:
                        route.append(cities[i])
                    # save bssf solution
                    bssf['soln'] = TSPSolution(route)
                else:
                    # increment prune number if lower bound is not better than bssf
                    pruned_states += 1
            else:
                # con
                children = cur_branch.get_children()
                total_states += len(children)
                for i in children:
                    # Time Complexity: O(NlogN) where N is the number of elements in the list
                    # Space Complexity: O(N)
                    heapq.heappush(queue, i)

        end_time = time.time()
        bssf['pruned'] = pruned_states  # Pruned states number
        bssf['max'] = max_queue_length  # Max queue size
        bssf['time'] = end_time - start_time  # Time spent to find bssf
        bssf['total'] = total_states  # Number of states created
        bssf['count'] = number_of_solutions  # Number solutions found during search

        return bssf

    # Helper function to quickly create and prepare the matrix needed for states

    # Time Complexity: O(n^2), let me explain why. First we have to create the matrix which takes O(n^2) time,
    # because the numpy library is used to create the matrix. Which takes O(n^2) time.
    # Then we have to fill the matrix with all zeros

    # Space Complexity: O(n^2), because we are creating a 2D array of size n^2 using the numpy library which is
    # filled up with all zeros
    def create_matrix(self):
        cities = self._scenario.getCities()
        city_length = len(cities)
        matrix = np.zeros((city_length, city_length))  # Time Complexity: O(n^2), Space Complexity: O(n^2)

        # Time Complexity: O(n^2) for both the outer and inner loop, but for space complexity we
        # are overwriting the same matrix created from the "numpy zeros functions" and using the
        # same space, so no additional space is used
        for i in cities:
            for j in cities:
                if i._index == j._index:
                    matrix[i._index][j._index] = np.inf
                else:
                    cost = i.costTo(j)
                    matrix[i._index][j._index] = cost
        return matrix

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

    def fancy(self, time_allowance=60.0):
        # Cheapest Insertion
        results = {}
        cities = self._scenario.getCities()  # Space Complexity: O(n) because we are storing all the cities in a list
        paths = self._scenario._edge_exists  # Space Complexity: O(n^2) because we are storing all the paths in a 2D array
        num_cities = len(cities)
        bssf = None

        start_time = time.time()
        for outer in range(num_cities):
            city_one = cities[outer]
            city_two = cities[outer]

            solved_path = [city_one]  # Start with the first city
            total_cost = 0

            # start_time = time.time()

            for i in range(num_cities):
                current_paths_city_one = paths[cities.index(city_one)]  # Time Complexity: O(1) because we are accessing the 2D array
                current_paths_city_two = paths[cities.index(city_two)]

                shortest_path = float('inf')  # Time Complexity: O(1)
                closest_city = None  # Time Complexity: O(1)
                fromCityOne = False

                for j in range(num_cities):  # Time Complexity: O(n)

                    if cities[j] not in solved_path:
                        if current_paths_city_one[j]:
                            distance1 = cities[j].costTo(city_one)  # Time Complexity: O(1)
                            # reverse_distance1 = cities[j].costTo(city_one)  # Time Complexity: O(1)
                            if distance1 < shortest_path:
                                shortest_path = distance1
                                closest_city = cities[j]  # Time Complexity: O(1)
                                fromCityOne = True
                        if current_paths_city_two[j]:
                            distance2 = city_two.costTo(cities[j])  # Time Complexity: O(1)
                            # reverse_distance2 = cities[j].costTo(city_two)  # Time Complexity: O(1)
                            if distance2 < shortest_path:
                                shortest_path = distance2
                                closest_city = cities[j]  # Time Complexity: O(1)
                                fromCityOne = False

                if closest_city is not None:

                    if fromCityOne:
                        city_one = closest_city
                        solved_path.insert(0, closest_city)  # Time Complexity: O(1)
                    else:
                        city_two = closest_city
                        solved_path.append(closest_city)  # Time Complexity: O(1)

                    total_cost += shortest_path
            if not (solved_path[-1].costTo(solved_path[0]) == float('inf')):
                # solved_path += solved_path[-1].costTo(solved_path[0])

                tmpBssf = TSPSolution(solved_path)  # Time Complexity: O(n), because it adds up the cost of the route

                if bssf is None or tmpBssf.cost < bssf.cost:
                    bssf = tmpBssf


        end_time = time.time()

        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = 1
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results



        pass
