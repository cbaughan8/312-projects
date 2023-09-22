from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF, QObject
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF, QObject
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))



import time
import node

# Some global color constants that might be useful
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

# Global variable that controls the speed of the recursion automation, in seconds
#
PAUSE = 0.25

#
# This is the class you have to complete.
#
class ConvexHullSolver(QObject):

# Class constructor
	def __init__( self):
		super().__init__()
		self.pause = False
		
# Some helper methods that make calls to the GUI, allowing us to send updates
# to be displayed.

	def showTangent(self, line, color):
		self.view.addLines(line,color)
		if self.pause:
			time.sleep(PAUSE)

	def eraseTangent(self, line):
		self.view.clearLines(line)

	def blinkTangent(self,line,color):
		self.showTangent(line,color)
		self.eraseTangent(line)

	def showHull(self, polygon, color):
		self.view.addLines(polygon,color)
		if self.pause:
			time.sleep(PAUSE)
		
	def eraseHull(self,polygon):
		self.view.clearLines(polygon)
		
	def showText(self,text):
		self.view.displayStatusText(text)

	# merge sort algorithm to sort the points by x value
	def sortByX(self, points):
		# splits the array into halves and recurse
		n = len(points)
		m = n//2
		if n == 1:
			return points
		left = points[:m]
		right = points[m:]
		self.sortByX(left)
		self.sortByX(right)

		# adds the lowest of either left or right to the sorted array until sorted
		i = j = k = 0
		while i < len(left) and j < len(right):
			if left[i].x() <= right[j].x():
				points[k] = left[i]
				i += 1
			else:
				points[k] = right[j]
				j += 1
			k += 1

		while i < len(left):
			points[k] = left[i]
			i += 1
			k += 1

		while j < len(right):
			points[k] = right[j]
			j += 1
			k += 1
		return points

	# finds the upper tangent by using the leftmost hull on the right and rightmost hull of the left
	def upperTangent(self, left, right):
		q = right
		p = left
		#finds the rightmost hull of the left
		while p.c_clockwise.point.x() > p.point.x():
			p = p.c_clockwise
		done = False
		# repeats until the upper tangent is found, if neither point changes, the tangent is found
		while not done:
			done = True
			# Finds the tangent by comparing the slopes of the left to right,
			# moving counterclockwise on the left point until the smallest slope is found.
			isTangent = (q.point.y() - p.point.y()) / (q.point.x() - p.point.x()) \
						>= (q.point.y() - p.c_clockwise.point.y()) / (q.point.x() - p.c_clockwise.point.x())
			while not isTangent:
				p = p.c_clockwise
				isTangent = (q.point.y() - p.point.y()) / (q.point.x() - p.point.x()) \
							>= (q.point.y() - p.c_clockwise.point.y()) / (q.point.x() - p.c_clockwise.point.x())
				done = False

			# Does the same as above, except moves the right point clockwise
			# until the largest slope is found
			isTangent = (q.point.y() - p.point.y()) / (q.point.x() - p.point.x()) \
						<= (q.clockwise.point.y() - p.point.y()) / (q.clockwise.point.x() - p.point.x())
			while not isTangent:
				q = q.clockwise
				isTangent = (q.point.y() - p.point.y()) / (q.point.x() - p.point.x()) \
							<= (q.clockwise.point.y() - p.point.y()) / (q.clockwise.point.x() - p.point.x())
				done = False

		return p,q


	# like upper tangent, but the slope comparisons are reversed
	def lowerTangent(self, left, right):
		q = right
		p = left
		while p.c_clockwise.point.x() > p.point.x():
			p = p.c_clockwise
		done = False
		while not done:
			done = True
			isTangent = (q.point.y() - p.point.y()) / (q.point.x() - p.point.x()) \
						<= (q.point.y() - p.clockwise.point.y()) / (q.point.x() - p.clockwise.point.x())
			while not isTangent:
				p = p.clockwise
				isTangent = (q.point.y() - p.point.y()) / (q.point.x() - p.point.x()) \
							<= (q.point.y() - p.clockwise.point.y()) / (q.point.x() - p.clockwise.point.x())
				done = False
			isTangent = (q.point.y() - p.point.y()) / (q.point.x() - p.point.x()) \
						>= (q.c_clockwise.point.y() - p.point.y()) / (q.c_clockwise.point.x() - p.point.x())
			while not isTangent:
				q = q.c_clockwise
				isTangent = (q.point.y() - p.point.y()) / (q.point.x() - p.point.x()) \
							>= (q.c_clockwise.point.y() - p.point.y()) / (q.c_clockwise.point.x() - p.point.x())
				done = False

		return p,q

	# This initializes a node for the linked list. It points the clockwise and counterclockwise
	# pointers to itself because it is the only node in the hull when initialized.
	def makeNode(self, point):
		n = node.node(None, None, point)
		n.clockwise = n
		n.c_clockwise = n
		return n

	# Connects the hulls by connecting the tangents. The pointers on the nodes of each tangent point are connected.
	def merge(self, left, right):
		# finds the points associated with each tangent
		left_upper, right_upper = self.upperTangent(left, right)
		left_lower, right_lower = self.lowerTangent(left, right)

		# connects the tangent points
		left_upper.clockwise = right_upper
		right_upper.c_clockwise = left_upper
		left_lower.c_clockwise = right_lower
		right_lower.clockwise = left_lower
		return left

	# Splits the starting into 2 halves and recursively calls itself until it
	# has split into all one node. It then calls merge on each pair of left and right nodes
	# until it has merged into one convex hull
	def divideAndConquer(self, points):
		# If the array size is 1, make a hull with one point
		if len(points) == 1:
			return self.makeNode(points[0])
		# split the array
		m = len(points)//2
		l_arr = points[:m]
		r_arr = points[m:]
		# recursive DC call
		l_list = self.divideAndConquer(l_arr)
		r_list = self.divideAndConquer(r_arr)
		# merge
		combined_list = self.merge(l_list, r_list)
		return combined_list

	# makes an array of lines for the GUI to draw
	def makePolygon(self, left):
		temp = left
		lines = []
		complete = False
		while not complete:
			if temp.c_clockwise == left:
				complete = True
			lines.append(QLineF(temp.point, temp.c_clockwise.point))
			temp = temp.c_clockwise
		return lines
	

# This is the method that gets called by the GUI and actually executes
# the finding of the hull
	def compute_hull( self, points, pause, view):
		self.pause = pause
		self.view = view
		assert( type(points) == list and type(points[0]) == QPointF )

		t1 = time.time()
		points = self.sortByX(points)
		t2 = time.time()

		t3 = time.time()
		points_list = self.divideAndConquer(points)
		polygon = self.makePolygon(points_list)
		t4 = time.time()

		# when passing lines to the display, pass a list of QLineF objects.  Each QLineF
		# object can be created with two QPointF objects corresponding to the endpoints
		self.showHull(polygon,RED)
		self.showText('Time Elapsed (Convex Hull): {:3.3f} sec'.format(t4-t3))



