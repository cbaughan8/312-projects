import math

# Number of child branches coming from each node in heap. If BRANCHES == 2, it is a binary heap
BRANCHES = 2

class HeapPQ:
    heapTree = []
    dist = {}
    positions = {}

    # I used the constructor for makeQueue()
    def __init__(self, nodes, dist):
        self.dist = dist
        for node in nodes:
            self.insert(node)

    # deletes the min and returns that min. Also rearranges heap back to min heap
    def deleteMin(self):
        # deletes the min node, stores it, and puts the last element on the top of the heap. Pops off the last index to decrease size
        ret_node = self.heapTree[0]
        self.heapTree[0] = self.heapTree[len(self.heapTree) - 1]
        node = self.heapTree[len(self.heapTree) - 1]
        self.positions[self.heapTree[0]] = 0
        self.heapTree.pop(len(self.heapTree) - 1)

        # moves the new top to where it belongs in the heap
        min_child = self.getSmallestChild(node)
        while not (min_child is None) and self.dist[node] > self.dist[min_child]:
            self.swap(min_child, node)
            min_child = self.getSmallestChild(node)
        return ret_node

    # decreases the distance of a certain node
    def decreaseKey(self, node, distance):
        # change the recorded distance
        self.dist[node] = distance
        # move the node to its new place in the min heap
        parent = self.getParent(node)
        while self.dist[node] < self.dist[parent] and self.positions[node] != 0:
            self.swap(node, parent)
            parent = self.getParent(node)

    # inserts new node into heap. Swaps new node and its parent until node is in the right spot
    def insert(self, node):
        self.heapTree.append(node)
        self.positions[node] = len(self.heapTree) - 1
        parent = self.getParent(node)
        while self.positions[node] > 0 and self.dist[parent] > self.dist[node]:
            self.swap(node, parent)
            parent = self.getParent(node)

    # finds the parent node of the current node
    def getParent(self, cur_node):
        global BRANCHES
        cur_index = self.positions[cur_node]
        return self.heapTree[(cur_index - 1) // BRANCHES]


    # finds the smallest child of the given node
    def getSmallestChild(self, node):
        # BRANCHES is the number of branches at each layer of the heap, it is set to 2 to be a binary heap
        global BRANCHES
        m = math.inf
        children = []

        # Return none if the child has no children
        if self.positions[node] >= len(self.heapTree) // BRANCHES:
            return None
        num_children = BRANCHES

        # finds all the node's children
        for i in range(BRANCHES):
            child_pos = self.positions[node] * BRANCHES + i + 1
            children.append(self.heapTree[child_pos])
            if child_pos == len(self.heapTree) - 1:
                num_children = i + 1
                break

        # finds the min child of the node
        min_node = None
        for i in range(num_children):
            if m >= self.dist[children[i]]:
                m = self.dist[children[i]]
                min_node = children[i]
        return min_node

    # swaps the positions of a child and parent node
    def swap(self, child, parent):
        self.heapTree[self.positions[child]] = parent
        self.heapTree[self.positions[parent]] = child

        temp = self.positions[child]
        self.positions[child] = self.positions[parent]
        self.positions[parent] = temp

    # returns the size of the queue
    def size(self):
        return len(self.heapTree)

