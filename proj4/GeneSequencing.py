#!/usr/bin/python3

from which_pyqt import PYQT_VER
from Cell import *
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import math
import time
import random

# Used to compute the bandwidth for banded version
MAXINDELS = 3

# Used to implement Needleman-Wunsch scoring
MATCH = -3
INDEL = 5
SUB = 1

class GeneSequencing:

	def __init__( self ):
		pass


	def unrestricted(self, seq1, seq2, align_length):
		# Cut string to align_length
		n = align_length if len(seq1) > align_length else len(seq1)
		m = align_length if len(seq2) > align_length else len(seq2)
		dp = [[None for j in range(m + 1)] for i in range(n + 1)]
		# Initialize first row and column
		for j in range(m + 1):
			dp[0][j] = Cell(j * 5, 0 , j - 1, 0, j)
		for i in range(n + 1):
			dp[i][0] = Cell(i * 5, i - 1, 0, i, 0)
		dp[0][0] = Cell(0, None, None, 0, 0)
		# Calculate cost of each cell
		for i in range (1, n + 1):
			for j in range (1, m + 1):
				cost_left = 5 + dp[i][j - 1].val
				cost_up = 5 + dp[i - 1][j].val
				cost_diag = (-3 if seq1[i - 1] == seq2[j - 1] else 1) + dp[i - 1][j - 1].val
				min_cost = min(cost_left, cost_up, cost_diag)
				if min_cost == cost_left:
					dp[i][j] = Cell(min_cost, i, j - 1, i, j)
				elif min_cost == cost_up:
					dp[i][j] = Cell(min_cost, i - 1, j, i, j)
				else:
					dp[i][j] = Cell(min_cost, i - 1, j - 1, i, j)
		# Set score to last cell
		score = dp[n][m].val
		alignment1 = ""
		alignment2 = ""
		cur_cell = dp[n][m]
		# Make alignments based on previous indices
		while cur_cell.prev_i is not None and cur_cell.prev_j is not None:
			if cur_cell.prev_i != cur_cell.i and cur_cell.prev_j != cur_cell.j:
				alignment1 = seq1[cur_cell.i - 1] + alignment1
				alignment2 = seq2[cur_cell.j - 1] + alignment2
			elif cur_cell.j != cur_cell.prev_j:
				alignment1 = "-" + alignment1
				alignment2 = seq2[cur_cell.j - 1] + alignment2
			else:
				alignment1 = seq1[cur_cell.i - 1] + alignment1
				alignment2 = "-" + alignment2
			cur_cell = dp[cur_cell.prev_i][cur_cell.prev_j]

		return score, alignment1, alignment2

	def banded_alg(self, seq1, seq2, align_length):
		# d == MININDELS, k == bandwidth
		d = MAXINDELS
		k = 7
		# Cut string to align_length
		s1_len = min(align_length, len(seq1))
		s2_len = min(align_length, len(seq2))
		# Return impossible for sequence lengths that don't align within the bandwidth
		if abs(s1_len - s2_len) > d:
			return math.inf, "No alignment possible", "No alignment possible"
		n = min(s1_len, s2_len)
		m = max(s1_len, s2_len)
		s1_is_short = False
		short_s, long_s = "",""
		# Find which sequence is longer
		if n == s1_len:
			short_s = seq1
			long_s = seq2
			s1_is_short = True
		else:
			short_s = seq2
			long_s = seq1
		dp = [[None for j in range(k)] for i in range(n+1)]
		# Calculate cost of each cell
		for i in range(n + 1):
			for j in range(k):
				# If cell is out of banded bounds, ignore
				if i + j < d or i + j - d - 1 >= m:
					continue
				# Initialize first cell
				if i == 0 and j == d:
					dp[i][j] = Cell(0, None, None, i, j)
					continue
				cost_left = math.inf
				cost_up = math.inf
				cost_diag = math.inf
				# Compute cost_left, cost_up, and cost_diag
				if j - 1 >= 0 and dp[i][j-1] is not None:
					cost_left = 5 + dp[i][j-1].val
				if i - 1 >= 0 and j + 1 < k and dp[i-1][j+1] is not None:
					cost_up = 5 + dp[i-1][j+1].val
				if i - 1 >= 0 and dp [i-1][j] is not None:
					cost_diag = -3 if short_s[i-1] == long_s[i + j - 4] else 1
					cost_diag += dp[i-1][j].val
				min_cost = min(cost_left, cost_up, cost_diag)
				if min_cost == cost_left:
					dp[i][j] = Cell(min_cost, i, j - 1, i, j)
				elif min_cost == cost_up:
					dp[i][j] = Cell(min_cost, i-1, j+1, i, j)
				else:
					dp[i][j] = Cell(min_cost, i-1, j, i, j)
		end_cell = None
		# Find end_cell and score
		for j in range(k - 1, -1, -1):
			if dp[n][j] is not None:
				end_cell = dp[n][j]
				break
		score = end_cell.val
		alignment_short = ""
		alignment_long = ""
		curr = end_cell
		# Finds the alignments of both the sequences
		while curr.prev_i is not None and curr.prev_j is not None:
			if curr.prev_i != curr.i and curr.prev_j != curr.j:
				alignment_short = short_s[curr.i - 1] + alignment_short
				alignment_long = "-" + alignment_long
			elif curr.prev_i != curr.i:
				alignment_short = short_s[curr.i - 1] + alignment_short
				alignment_long = long_s[curr.i + curr.j - 4] + alignment_long
			else:
				alignment_short = "-" + alignment_short
				alignment_long = long_s[curr.i + curr.j - 4] + alignment_long
			curr = dp[curr.prev_i][curr.prev_j]
		# changes the order to return the alignment in the right place
		if s1_is_short:
			return score, alignment_short, alignment_long
		else:
			return score, alignment_long, alignment_short

	
# This is the method called by the GUI.  _seq1_ and _seq2_ are two sequences to be aligned, _banded_ is a boolean that tells
# you whether you should compute a banded alignment or full alignment, and _align_length_ tells you 
# how many base pairs to use in computing the alignment

	def align( self, seq1, seq2, banded, align_length):
		self.banded = banded
		self.MaxCharactersToAlign = align_length
		score, alignment1, alignment2 = "","",""
		if banded:
			score, alignment1, alignment2 = self.banded_alg(seq1, seq2, align_length)
		else:
			score, alignment1, alignment2 = self.unrestricted(seq1, seq2, align_length)

		alignment1 = alignment1[:100]
		alignment2 = alignment2[:100]


###################################################################################################
# your code should replace these three statements and populate the three variables: score, alignment1 and alignment2
# 		score = random.random()*100;
# 		alignment1 = 'abc-easy  DEBUG:({} chars,align_len={}{})'.format(
# 			len(seq1), align_length, ',BANDED' if banded else '')
# 		alignment2 = 'as-123--  DEBUG:({} chars,align_len={}{})'.format(
# 			len(seq2), align_length, ',BANDED' if banded else '')
###################################################################################################					
		
		return {'align_cost':score, 'seqi_first100':alignment1, 'seqj_first100':alignment2}


