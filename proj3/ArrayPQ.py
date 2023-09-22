import math

class ArrayPQ:
    queue = {}

    # I used the constructor for makeQueue()
    def __init__(self, nodes, dist):
        for u in nodes:
            self.queue[u] = dist[u]

    # deletes the min in the queue and returns that min
    def deleteMin(self):
        m = math.inf
        ret = None
        for key in self.queue.keys():
            if m >= self.queue[key]:
                m = self.queue[key]
                ret = key
        self.queue.pop(ret)
        return ret

    # decreases the distance of a node
    def decreaseKey(self, node, val):
        self.queue[node] = val

    # inserts a node in the queue
    def insert(self, node, val):
        self.queue[node] = val

    # returns the queue size
    def size(self):
        return len(self.queue)


