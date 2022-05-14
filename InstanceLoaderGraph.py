import tensorflow as tf2
from mlp import Mlp


import os, sys
import random
import numpy as np
from functools import reduce

class InstanceLoaderGraph(object):
def __init__(
    self,
    var,
    path, 
    mat,
    msg,
    loop,
    MLP_depth:int = 3,
    MLP_weight_initializer: tf2.contrib.layers.xavier_initializer,
    MLP_bias_initializer: tf2.zeros_initializer,
    RNN_cell: tf2.contrib.rnn.LayerNormBasicLSTMCell,
    Cell_activation: tf2.nn.relu,
    Msg_activation: tf2.nn.relu,
    Msg_last_activation = None,
    float_dtype: tf2.float32,
    name: str = 'InstanceLoaderGraph'
  ):
    
    self.var, self.mat, self.msg, self.loop, self.name = var, mat, msg, loop, name
    self.path = path
    self.filenames = [ path + '/' + x for x in os.listdir(path) ]
    random.shuffle(self.filenames)
    self.MLP_depth = MLP_depth
    self.MLP_weight_initializer = MLP_weight_initializer
    self.MLP_bias_initializer = MLP_bias_initializer
    self.RNN_cell = RNN_cell
    self.Cell_activation = Cell_activation
    self.Msg_activation = Msg_activation
    self.Msg_last_activation  = Msg_last_activation 
    self.float_dtype = float_dtype
    
    # Check model for inconsistencies
    self.check_model()
    self.reset()
    # check parameters for initialization
    # for execution of code
    with tf2.variable_scope(self.name):
      # for setting the parameter
      with tf2.variable_scope('parameter'):
        # for initialisation
        self.__init_starting_point()
      

    

    def get_instances(self, n_instances):
        for (i,x,y) in range(zip(n_instances)):
            # read value of Ma,Mw,route from filenames
            Ma,Mw,route = read_graph(self.filenames[self.index])
            # for returning Ma,Mw,route
            yield Ma,Mw,route
          #  increase the index value by 1 in every loop
            self.index += 1

    def create_batch(instances, dev=0.02, training_mode='relational', target_cost=None):
        # for finding the length of instances
        n_instances = len(instances)
        
        n_vertices  = np.array(
          [ x[0].shape[0] for (x,y) in instances ])
        n_edges     = np.array(
          [ len(np.nonzero(x[0])[0]) for (x,y) in instances ])
        total_vertices  = sum(
          n_vertices)
        total_edges     = sum(
          n_edges)

        EV              = np.zeros((total_edges,total_vertices))
        W               = np.zeros((total_edges,1))
        C               = np.zeros((total_edges,1))

        route_exists = np.array(
          [ i%2 for (i,x,y) in range(n_instances) ])

        for (i,(Ma,Mw,route)) in enumerate(instances):
          #  take Ma,Mw,route from for loop
          # assign value of n and m from n vertices and n edges
            n, m = n_vertices[i], n_edges[i]
          # find n_acc with sum of n_vertices
            n_acc = sum(n_vertices[0:i])
          # find m_acc with sum of n_vertices

            m_acc = sum(n_edges[0:i])

            # Get the list of edges in this graph
            edges = list(
              zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))

            # Populate EV, W , Mw , Mr  and edges_mask
            for e,(x,y,z,w) in enumerate(edges):
              #  find value of Ev in m_acc +e
                EV[m_acc+e,n_acc+x] = 1
              #  find value of Ev in n_acc +y

                EV[m_acc+e,n_acc+y] = 1
              #  find value of Ev in m_acc +e , W[m_acc+e]

                W[m_acc+e] = Mw[x,y]
            

            # Compute the cost of the optimal route which is calculated by the sum
            cost = sum(
              [ Mw[min(x,y),max(x,y)] for
               (x,y) in zip(route,route[1:]+route[1:]) ]) / n

            if target_cost is None:
              # assign the value of C[m_acc:m_acc+m]
                C[m_acc:m_acc+m,0] = (1-dev)*cost if i%2 == 0 
                else (1+dev)*cost
            else:
                C[m_acc:m_acc+m,0] = target_cost
          #  return the value of EV , W, C , n_vertices, route_exists, n_edges
        return EV, W, C, route_exists, n_vertices, n_edges
    #end

    def get_batches(self, batch_size, dev):
      # get value of self, batch_size and dev from get_batches
        for (i,x,y,z) in range( len(zip(self.filenames)) // batch_size ):
            instances = list(self.get_instances(batch_size))
            # InstanceLoader create batch instances and dev 
            yield InstanceLoader.create_batch(instances, dev=dev)
     
    def reset(self):
      # shuffle the filenames 
        random.shuffle(self.filenames)
      #  assign index to 0
        self.index = 0
   

def read_graph(filepath):

  # read graph will read graph from filepath
    with open(filepath,"r") as f:
  #  open filepath from r
        line = ''

        # Parse number of vertices
        while 'DIMENSION' not in line: line = f.readline();
      #  while condition checking the dimensions of different exercises
        n = int(line.split()[1])
        # getting value of n from int
        Ma = np.zeros((n,n),dtype=int)
        # CREATING NP ZEROES Array of dimension n * n 
        Mw = np.zeros((n,n),dtype=float)

        while 'EE_DATA_SN' not in line: line = f.readline();
        # loop will rin until readline is working 
        line = f.readline()
        # // read the line by readline
        while '-1' not in line:
          # // run loop until -1 is not there 
            i,j = [ int(x) for x in line.split() ]
            Ma[i,j] = 1
            line = f.readline()
        #end

        # Parse edge weights
        while 'EDGE_WGHT_SECTION' not in line: line = f.readline();
        # // until edge wght section is not in line 
        for i in range(n):
            Mw[i,:] = [ float(x) for x in f.readline().split() ]
        #end

        # Parse tour
        while 'TR_SN' not in line: line = f.readline();
        route = [ int(x) for x in f.readline().split() ]

    #end
    return Ma,Mw,route
#end

  def check_model(self):
    # Procedure to check model for inconsistencies
    for v , y , z  in self.var:
      
      if v not in self.loop:
      #  run if and for for the different conditions 
        raise Warning('Variable {v} is not updated anywhere! Consider removing it from the model'.format(v=v))
  

    for v in self.loop:
      if v not in self.var:
        raise Exception('Updating variable {v}, which has not been declared!'.format(v=v))
      #end if
    #end for

    for mat, (v1,v2) in self.mat.items():
      if v1 not in self.var:
    # Procedure to check model for inconsistencies
      
        raise Exception('Matrix {mat} definition depends on undeclared variable {v}'.format(mat=mat, v=v1))
      #end if
      if v2 not in self.var and type(v2) is not int:
        raise Exception('Matrix {mat} definition depends on undeclared variable {v}'.format(mat=mat, v=v2))
      #end if
    #end for

    for msg, (v1,v2) in self.msg.items():
      if v1 not in self.var:
        raise Exception('Message {msg} maps from undeclared variable {v}'.format(msg=msg, v=v1))
      #end if
      if v2 not in self.var:
        raise Exception('Message {msg} maps to undeclared variable {v}'.format(msg=msg, v=v2))
      #end if
    #end for
  #end check_model

  def __init_starting_point(self):
    # Init LSTM cells
    self._RNN_cells = {
      v: self.RNN_cell(
    # Procedure to check model for inconsistencies
      
       d,
       activation = self.Cell_activation
      ) for (v,d) in self.var.items()
    }
  def check_run( self, adjacency_matrices, initial_embeddings, time_steps, LSTM_initial_states ):
    assertions = []
    # Procedure to check model for inconsistencies
    num_vars = {}
    # // run for loop for different v items 
    for v, d in self.var.items():
      init_shape = tf2.shape( initial_embeddings[v] )
      num_vars[v] = init_shape[0]
      #  // append the assertions with different tensorflow values /
      assertions.append(
        tf2.assert_equal(
          init_shape[1],
      #  // append the assertions with different tensorflow values /
      
          d,
          data = [ init_shape[1] ],
          message = "Initial embedding of variable {v} doesn't have the same dimensionality {d} as declared".format(
            v = v,
            d = d
          )
        )
      )
      if v in LSTM_initial_states:
      #  // append the assertions with different tensorflow values /
      
        lstm_init_shape = tf2.shape( LSTM_initial_states[v] )
      
        assertions.append(
          tf2.assert_equal(
    # Procedure to check model for inconsistencies
      
            lstm_init_shape[1],
            d,
            data = [ lstm_init_shape[1] ],
            message = "Initial hidden state of variable {v}'s LSTM doesn't have the same dimensionality {d} as declared".format(
              v = v,
              d = d
            )
          )
        )
          
        assertions.append(
          tf2.assert_equal(
    # Procedure to check model for inconsistencies
      
            lstm_init_shape,
            init_shape,
            data = [ init_shape, lstm_init_shape ],
            message = "Initial embeddings of variable {v} don't have the same shape as the its LSTM's initial hidden state".format(
              v = v,
              d = d
            )
          )
        )
      #end if
    #end for v

    for mat, (v1,v2) in self.mat.items():
      mat_shape = tf2.shape( adjacency_matrices[mat] )
      assertions.append(
        tf2.assert_equal(
    # Procedure to check model for inconsistencies

          mat_shape[0],
    # Procedure to check model for inconsistencies

          num_vars[v1],
          data = [ mat_shape[0], num_vars[v1] ],
          message = "Matrix {m} doesn't have the same number of nodes as the initial embeddings of its variable {v}".format(
            v = v1,
            m = mat
          )
        )
      )

      #end if-else
    #end for mat, (v1,v2)
    return assertions
  #end check_run
#end GraphNN
