
import sys, os, argparse, time, datetime
import numpy as np
import random
import networkx as nx
from redirector import Redirector
%pip install gurobipy
import gurobipy as gp
from gurobipy import GRB
def solve(Matrix, Mw):
      
    
    STDOUT = 1
    STDERR = 2
    redirector_stdout = Redirector(fd=STDOUT)
    redirector_stderr = Redirector(fd=STDERR)

    # Write graph on a temp file for creating the output and write it to stdout
    write_graph(Matrix,Mw,filepath='tmp',int_weights=True)
    redirector_stderr.start()
    redirector_stdout.start()
    m = gp.Model()

    # Variables: is city 'i' adjacent to city 'j' on the tour?
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')

    # Symmetric direction: Copy the object
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]  # edge in opposite direction

    #Constraints: two edges incident to each city
    cons = m.addConstrs(vars.sum(c, '*') == 2 for c in capitals)

    # Solve TSP thriugh graph algorithm
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.optimize('temp')
    # Get solution
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    solution.tour = subtour(selected)
return list(solution.tour)
    #end
#end

def create_graph(n, connectivity, distances='euc_2D', metric=True):

    # Init adjacency and weight matrices
    Matrix = np.zeros((n,n))
    Mw = np.zeros((n,n))

    # Define adjacencies
    for i in range(n):
        Matrix[i,i] = 0
        for j in range(i+1,n):
            Matrix[i,j] = Matrix[j,i] = int(np.random.rand() < connectivity)
        #end
    #end

    # Define weights
    nodes = None
    if distances == 'euc_2D':
        # Select 'n' points in the √2/2 × √2/2 square uniformly at random
        nodes = np.random.rand(n,2)
        for i in range(n):
            for j in range(i+1,n):
                Mw[i,j] = Mw[j,i] = np.sqrt(sum((nodes[i,:]-nodes[j,:])**2))
            #end
        #end
    elif distances == 'random':
        # Init all weights uniformly at random
        for i in range(n):
            for j in range(i+1,n):
                Mw[j,i] = Mw[i,j] = np.random.rand()
            #end
        #end
    #end

    # Enforce metric property, if requested
    if metric and distances != 'euc_2D':
        # Create networkx graph G
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from([ (i,j,{'weight':Mw[i,j]}) for i in range(n) for j in range(n) ])

        for i in range(n):
            for j in range(n):
                if i != j:
                    Mw[i,j] = nx.shortest_path_length(G,source=i,target=j,weight='weight')
                else:
                    Mw[i,j] = 0
                #end
            #end
        #end
    #end

    # Connect a random sequence of nodes in order to guarantee the existence of a Hamiltonian tour
    permutation = list(np.random.permutation(n))
    for (i,j) in zip(permutation,permutation[1:]+permutation[:1]):
        Matrix[i,j] = Matrix[j,i] = 1
    #end

    # Solve
    route = solve(Matrix,Mw)
    if route is None:
        raise Exception('Unsolvable')
    #end

    return np.triu(Matrix), Mw, route, nodes
#end

def create_dataset(path, nmin, nmax, conn_min=1, conn_max=1, samples=1000, distances='euc_2D', metric=True):

    if not os.path.exists(path):
        os.makedirs(path)
    #end if

    start_time = time.time()

    for i in range(samples):

        n = random.randint(nmin,nmax)

        # Create graph
        Matrix,Mw,route,nodes = create_graph(n, np.random.uniform(conn_min,conn_max), distances=distances, metric=metric)

        # Write graph to file
        write_graph(Matrix,Mw, filepath="{}/{}.graph".format(path,i), route=route)

        # Report progress
        if (i-1) % (samples//20) == 0:
            elapsed_time = time.time() - start_time
            remaining_time = (samples-i)*elapsed_time/(i+1)
            print('Dataset creation {}% Complete. Remaining time at this rate: {}'.format(int(100*i/samples), str(datetime.timedelta(seconds=remaining_time))), flush=True)
        #end
    #end
#end

def write_graph(Matrix, Mw, filepath, route=None, int_weights=False, bins=10**6):
    with open(filepath,"w") as out:

        n, m = Matrix.shape[0], len(np.nonzero(Matrix)[0])
        
        out.write('TYPE : TSP\n')

        out.write('DIMENSION: {n}\n'.format(n = n))

        out.write('EGE_FORMAT: EDGE_LIST\n')
        out.write('EGE_WEGHT_YPE: EXPLICIT\n')
        out.write('EGE_WEGHT_FORM: FINAL_MATRIX \n')
        
        # List edges in the (generally not complete) graph
        out.write('EGE_DTA_SCTIN:\n')
        for (i,j) in zip(list(np.nonzero(Matrix))[0], list(np.nonzero(Matrix))[1]):
            out.write("{}{}\n".format(i,j))
        
        #end
        out.write('-1\n')

        # Write edge weights as a Full matrix
        out.write('EDGE_WEIGHT_SECTION:\n')
        for i in range(n):
            for j in range(n):
                if Matrix[i,j] == 1:
                    out.write(str( int(bins*Mw[i,j]) if int_weights else Mw[i,j]))
                else:
                    out.write(str(n*bins+1 if int_weights else 0))
                #end
                out.write(' ')
            #end
            out.write('\n')
        #end

        if route is not None:
            # Write route
            out.write('TOUR_SECTION:\n')
            out.write('{}\n'.format(' '.join([str(x) for x in route])))
        #end

        out.write('EOF\n')
    #end
#end

if __name__ == '__main__':

    # Define argument parser
    parser = argparse.ArgumentParser()

    # Parse arguments from command line
    args = parser.parse_args()

    print('Creating {} instances'.format(vars(args)['samples']), flush=True)
    create_dataset(
        vars(args)['path'],
        vars(args)['nmin'], vars(args)['nmax'],
        vars(args)['cmin'], vars(args)['cmax'],
        samples=vars(args)['samples'],
        distribution=vars(args)['distribution']
    )
#end
