# Python Program to detect cycles in a directed graph
from collections import defaultdict
import re

INDEX = 0 
LOW_LINK = 1
ON_STACK = 2

INFINITE = float('inf') 

# This class represents a undirected
# graph using adjacency list representation

class Graph:
 
    def __init__(self):
        # Default dictionary to store graph
        self.edges = defaultdict(list)

        #List of vertices
        self.vertices = []

        self.currentComp = []
        self.connectedComps = []

    def searchVertices(self, pattern):
        result = []
        for v in self.vertices:
            idx = v.find(":-")
            if idx == -1:
                label = v
            else:
                label = v[:idx]
            regex = re.compile(pattern)
            if regex.search(label):
                result.append(v)
        return result
    
    def addVertex(self, v):
        if v not in self.vertices:
            self.vertices.append(v)

    # Function to add an edge to graph
    def addEdge(self, v, w, neg = False):
 
        # Add w to v_s list
        self.edges[v].append((w, neg))
        #self.edges[w].append((v, neg))
 
    # A recursive function that uses
    # visited[] and parent to detect
    # cycle in subgraph reachable from vertex v.
    def isCyclicUtil(self, v, visit, parent = ""):
 
        self.currentComp.append(v)
        # Mark the current node as visited
        visit[v] = True
 
        # Recur for all the vertices
        # adjacent to this vertex
        for vertex, weight in self.edges[v]:
            # Process only positive weighted edges
            if weight != -1: 
                # If the node is not
                # visited then recurse on it
                if not visit[vertex]:
                    if self.isCyclicUtil(vertex, visit, v):
                        return True
                # If an adjacent vertex is
                # visited and not parent
                # of current vertex,
                # then there is a cycle
                elif parent != '' and parent != vertex:
                    return True
                # Return True for self references
                elif vertex == v:
                    return True
            
        return False
 
    # Returns true if the graph
    # contains a positive cycle, else false.

    def isPositiveCyclic(self):
        # Mark all the vertices
        # as not visited
        visit = {k: False for k in self.vertices}

        # Call the recursive helper
        # function to detect a positive cycle in different
        # DFS trees
        for vertex, visited in visit.items():
            # Don't recur for u if it is already visited
            if not visited:
                if self.isCyclicUtil(vertex, visit):
                    return True

        return False

    def isNegCycleBellmanFord(self, src, dist, visit):
        V = len(self.vertices)

        # Step 1: Initialize distances from src
        # to all other vertices as INFINITE
        dist = {k: INFINITE for k in self.vertices}
        dist[src] = 0
    
        # Step 2: Relax all edges |V| - 1 times.
        # A simple shortest path from src to any
        # other vertex can have at-most |V| - 1
        # edges
        for i in range(1, V):
            for u, adjacencies in self.edges.items():
                for v, weight in adjacencies:
                    if (dist[u] != INFINITE and dist[u] + weight < dist[v]):
                        dist[v] = dist[u] + weight
                        if not visit[v]:
                            visit[v] = True
    
        # Step 3: check for negative-weight cycles.
        # The above step guarantees shortest distances
        # if graph doesn't contain negative weight cycle.
        # If we get a shorter path, then there
        # is a cycle.
        for u, adjacencies in self.edges.items():
            for v, weight in adjacencies:
                if (dist[u] != INFINITE and dist[u] + weight < dist[v]):
                    return True
    
        return False

    def isNegativeCyclic(self):
        # Initialize list of visited vertices
        # to all other vertices as INFINITE
        dist = {k: INFINITE for k in self.vertices}   
        visit =  {k: False for k in self.vertices}   

        # Call Bellman-Ford for all those vertices
        # that are not visited
        for vertex, visited in visit.items():
            if not visited:
                visited = True
                if (self.isNegCycleBellmanFord(vertex, dist, visit)):
                    return True 

        return False