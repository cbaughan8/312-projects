#!/usr/bin/python3

from CS312Graph import *
from ArrayPQ import *
from HeapPQ import *
import time


class NetworkRoutingSolver:

    dist = {}
    prev = {}

    def __init__( self):
        pass

    def initializeNetwork( self, network ):
        assert( type(network) == CS312Graph )
        self.network = network

    def getShortestPath( self, destIndex ):
        #initialize variables for the loop and update self.dest
        self.dest = destIndex
        path_edges = []
        node = self.network.nodes[destIndex]
        total_length = self.dist[node]
        start = self.network.nodes[self.source]
        edge = None
        dst = node
        src = self.prev[dst]

        # loops until a path to the start is found or there is no path
        while node != start and src is not None:
            for neighbor in src.neighbors:
                if neighbor.dest.node_id == dst.node_id:
                    edge = neighbor
                    break
            path_edges.append((edge.src.loc, edge.dest.loc, '{:.0f}'.format(edge.length)))
            node = src
            dst = node
            src = self.prev[dst]
        return {'cost':total_length, 'path':path_edges}

    def computeShortestPaths( self, srcIndex, use_heap=False ):
        self.source = srcIndex
        t1 = time.time()
        nodes = self.network.nodes
        self.dist, self.prev = self.dijkstras(nodes, srcIndex, use_heap)
        t2 = time.time()
        return (t2-t1)

    def dijkstras(self, nodes, src_index, use_heap):
        dist = {}
        prev = {}
        # set distances to infinity and previous nodes to none
        for u in nodes:
            dist[u] = math.inf
            prev[u] = None
        dist[nodes[src_index]] = 0.0
        # make either the array or heap queue
        Q = None
        if use_heap:
            Q = HeapPQ(nodes, dist)
        else:
            Q = ArrayPQ(nodes, dist)
        # find the distance from the start node to each node and save the previous node
        while Q.size() > 0:
            u = Q.deleteMin()
            for v in u.neighbors:
                n = v.dest.node_id
                if dist[nodes[n]] > dist[u] + v.length:
                    dist[nodes[n]] = dist[u] + v.length #check later
                    prev[nodes[n]] = u
                    Q.decreaseKey(nodes[n], dist[nodes[n]]) #check
        return dist, prev






