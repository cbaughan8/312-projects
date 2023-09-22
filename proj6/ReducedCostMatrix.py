# Helper class for Branch and Bound algorithm.
# This reduced cost matrix data structure helps keep track of lower bounds and whether locations have been visited.
import numpy as np


class ReducedCostMatrix:
    # Initialization/updating row and col
    # Time complexity: O(n^2), let me explain, we have to iterate through the entire matrix to find the
    # current minimum value in each row and column. Since we have two separate for loops, the time complexity for
    # each for loop is O(n^2). Therefore, the time complexity for the entire function is O(n^2) + O(n^2) = O(n^2).

    # Space complexity: O(n^2), let me explain, we have to account the space of the entire matrix which is of dimensions
    # (n x n). Since we have two separate for loops, the space complexity for each for loop is O(n^2). Therefore, the
    # space complexity for the entire function is O(n^2) + O(n^2) = O(n^2).

    def __init__(self, matrix, visited, lower_bound=0):
        self.matrix = matrix.copy()
        self.lower_bound = lower_bound
        self.visited = visited
        for row in range(len(matrix[:, 0])):
            min_val = np.min(matrix[row, :])  # Time Complexity: O(n), because it needs to loop for
                                              # the minimum of an array or minimum along an axis.
            if min_val != np.inf and min_val > 0:
                self.matrix[row, :] -= min_val
                self.lower_bound += min_val

        for col in range(len(matrix[0, :])):
            min_val = np.min(self.matrix[:, col])
            if min_val != np.inf and min_val > 0:
                self.matrix[:, col] -= min_val
                self.lower_bound += min_val

    # Returns the number of visited nodes in total
    # Time Complexity: O(1)
    # Space Complexity: O(1)
    def num_visited(self):
        return len(self.visited)

    # Returns the array of children
    # Time Complexity: O(n^3), let me explain, in this function, we have a for loop
    # which iterates the length of the matrix, which by itself would make this function
    # O(n). However, because within the for loop we call the ReducedCostMatrix, which has a
    # time complexity of O(n^2), then our get children function now has a time complexity
    # of O(n^2). This is assuming the numpy functions used are O(n) as well.

    # Space Complexity: O(n^3), let me explain. In this function we have a for loop
    # which iterates the length of the matrix and makes a copy of it, making it O(n).
    # However, because the ReducedCostMatrix has a space complexity of O(n^2) which
    # with the surrounding O(n) loop makes the space complexity O(n^3)

    def get_children(self):
        children = []
        # This line gets the current city by pulling the last city in visited
        current_city = self.visited[-1]
        # O(n) as it is a for loop that just loops through the size of the matrix
        for i in range(len(self.matrix[0, :])):
            if self.matrix[current_city, i] != np.inf:
                lower_bound = self.lower_bound + self.matrix[current_city, i]  # O(n)
                child_matrix = self.matrix.copy()  # O(n)
                child_matrix[current_city, :].fill(np.inf)  # O(n)
                child_matrix[:, i].fill(np.inf)  # O(n)
                child_matrix[i, current_city] = np.inf  # O(n)
                visited = self.visited.copy()
                visited.append(i)
                child = ReducedCostMatrix(child_matrix, visited, lower_bound=lower_bound)  # O(n^2)
                children.append(child)
        return children

    # Checks to see if you are able to keep going, basically checking that it is not infinite (visited)
    # Time Complexity: O(n)
    # Space Complexity: O(1)
    def path_to_home(self):
        return np.min(self.matrix[:, 0]) != np.inf

    def __lt__(self, other):
        if len(self.visited) != len(other.visited):
            return len(self.visited) > len(other.visited)
        return self.lower_bound < other.lower_bound

    def __str__(self):
        result = ('lower_bound: %s, len_visited: %s, current: %s\n' %
                  (self.lower_bound, len(self.visited), self.visited[-1]))
        for i in range(len(self.matrix[:, 0])):
            row = '[ '
            for j in range(len(self.matrix[0, :])):
                row += '{:^7}'.format(self.matrix[i, j])
            row += ' ]\n'
            result += row
        return result
