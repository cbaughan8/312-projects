#!/usr/bin/python3
import copy
import math

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools

START_CITY_INDEX = 0

# State used for branch and bound
class State:
	def __init__(self, matrix, path, bound, num_cities):
		self.matrix = matrix
		self.path = path
		self.bound = bound
		# used to determine which state to pop off the queue first
		self.priority =  bound - bound * len(path) / num_cities

	def __lt__(self, other):
		return self.priority < other.priority


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
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
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
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

	def greedy( self,time_allowance=60.0 ):
		# initialize variables for the result
		results = {}
		path = []
		count = 0
		cities = self._scenario.getCities()
		solution_found = False
		start_time = time.time()
		while (not solution_found) and time.time() - start_time < time_allowance:
			# loops through each city to try each one as a starting city
			for city in cities:
				path = [city]
				cities_left = copy.copy(cities)
				cities_left.remove(city)
				# The greedy loop, finds the minimum cost to a city until all cities are included
				# If a solution cannot be found, this loop is broken and another start city is tried
				while len(cities_left) > 0:
					cityToRemove = None
					min_cost = math.inf
					for city2 in cities_left:
						if city.costTo(city2) < min_cost:
							cityToRemove = city2
							min_cost = city.costTo(city2)
					if cityToRemove is None:
						break
					else:
						cities_left.remove(cityToRemove)
						path.append(cityToRemove)
				if len(cities_left) == 0 and path[-1].costTo(city) != math.inf:
					solution_found = True
					count += 1
					break
			if not solution_found:
				break
		# Results made for return value
		end_time = time.time()
		solution = TSPSolution(path)
		results['soln'] = solution
		results['cost'] = solution.cost if solution_found else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
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
		
	def branchAndBound( self, time_allowance=60.0 ):
		# Initialize variables needed for calculations and results
		global START_CITY_INDEX
		start_time = time.time()
		results = {}
		count = 0
		cities = self._scenario.getCities()
		num_cities = len(cities)
		maxQsize = 0
		bssf = None
		num_states = 0
		pruned_states = 0
		greedy_solution = self.greedy(1.0)['soln']
		# find starting solution, first using greedy, if no greedy found, use random solution
		# if no random or greedy solution found, start with cost at infinity
		if greedy_solution is not None and greedy_solution.cost != math.inf:
			bssf = greedy_solution
		else:
			bssf = self.defaultRandomTour(1.0)['soln']
		if bssf is None:
			bssf = TSPSolution([])
			bssf.cost = math.inf

		# Use costs to make the initial cost matrix
		initial_matrix = np.array([[math.inf for x in range(num_cities)] for y in range(num_cities)])
		for i in range(num_cities):
			for j in range(num_cities):
				initial_matrix[i,j] = cities[i].costTo(cities[j])
		initial_path = []
		initial_state = State(initial_matrix, initial_path, 0, len(cities))
		# find the starting bound and starting matrix
		start_matrix, start_bound = self.reduce_cost_matrix(initial_state)
		start_state = State(start_matrix, [cities[START_CITY_INDEX]], start_bound, len(cities))
		# make the Q
		Q = []
		tie_breaker = 1
		maxQsize = 1
		heapq.heappush(Q, (start_state, tie_breaker))
		# brand and bound loop, loops until the length of the Q is 0
		while time.time() - start_time < time_allowance and len(Q) > 0:
			cur_state = heapq.heappop(Q)[0]
			# prune states that have a bound greater than the best cost
			if cur_state.bound < bssf.cost:
				for i in range(len(cur_state.matrix[cur_state.path[-1]._index])):
					if cur_state.matrix[cur_state.path[-1]._index, i] != math.inf:
						num_states += 1
						# find the child state using current state and doing making the new reduced cost matrix
						child_state = self.find_lower_bound(cur_state, i)
						# Sets bssf to the new solution if there is one
						if len(child_state.path) == len(cities) and child_state.matrix[i, START_CITY_INDEX] != math.inf:
							count += 1
							solution = TSPSolution(child_state.path)
							if solution.cost < bssf.cost:
								bssf = solution
						# Puts child states on the queue if their cost is less than the solution cost
						elif child_state.bound < bssf.cost:
							tie_breaker += 1
							heapq.heappush(Q, (child_state, tie_breaker))
							maxQsize = max(maxQsize, len(Q))
						else:
							pruned_states += 1
			else:
				pruned_states += 1
		end_time = time.time()
		# Makes the result to return to the GUI
		if end_time - start_time > time_allowance:
			pruned_states += len(Q)
		if bssf.cost == math.inf:
			return None
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = maxQsize
		results['total'] = num_states
		results['pruned'] = pruned_states
		return results

	# Calculates the bound and cost matrix of a child state given a parent state
	def reduce_cost_matrix(self, parent_state):
		cities = self._scenario.getCities()
		new_matrix = copy.copy(parent_state.matrix)
		new_bound = parent_state.bound
		# row calculations
		for i in range(len(new_matrix)):
			row_min = min(new_matrix[i])
			if row_min != math.inf:
				new_matrix[i] = [x - row_min for x in new_matrix[i]]
				new_bound += row_min
			else:
				if cities[i] not in parent_state.path or i == parent_state.path[-1]._index:
					new_bound = math.inf
		# column calculations
		for j in range(len(new_matrix[0])):
			col_min = min(new_matrix[:, j])
			if col_min != math.inf:
				new_matrix[:, j] = [x - col_min for x in new_matrix[:, j]]
				new_bound += col_min
			else:
				if cities[j] not in parent_state.path or j == parent_state.path[0]._index:
					new_bound = math.inf
		return new_matrix, new_bound

	# Produces a child state by doing the appropriate calculations to the parent matrix, bound and path
	# Calls reduce_cost_matrix for part of the calculation
	def find_lower_bound(self, parent_state, child_index):
		cities = self._scenario.getCities()
		path_end = parent_state.path[-1]._index
		child_bound = parent_state.bound + parent_state.matrix[path_end, child_index]
		child_matrix = copy.copy(parent_state.matrix)
		# makes the appropriate row, column, and place opposite the starting place infinite
		child_matrix[path_end] = math.inf
		child_matrix[:, child_index] = math.inf
		child_matrix[child_index, path_end] = math.inf
		child_path = copy.copy(parent_state.path)
		child_path.append(cities[child_index])
		child_matrix, child_bound = self.reduce_cost_matrix(State(child_matrix, child_path, child_bound, len(cities)))
		return State(child_matrix, child_path, child_bound, len(cities))



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

		# current_city = cities[0]

		city_one = cities[0]
		city_two = cities[0]


		solved_path = [city_one]  # Start with the first city
		total_cost = 0
		start_time = time.time()

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
						reverse_distance2 = cities[j].costTo(city_two)  # Time Complexity: O(1)
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




