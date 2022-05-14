
import sys, os
import tensorflow as tf

from InstanceLoaderGraph import InstanceLoaderGraph



from mlp import Mlp


sys.path.insert(1, os.path.join(sys.path[0], '..'))

def build_dr(d):

   graph_initiation_edge_MLP = Mlp(
        layer_sizes = [ d/8, d/4, d/2 ],
        activations = [ tf.nn.relu for _ in range(3) ],
        output_size = d,
        name = 'E_init_MLP',
        name_internal_layers = True,
        kernel_initializer = tf.xavier_initializer(),
        bias_initializer = tf.zeros_initializer()
    )


    # Define hyperparameters
    d = d
    learning_rate = 2e-5
    l2norm_scaling = 1e-10
    global_norm_gradient_clipping_ratio = 0.65

    route_exists = tf.placeholder( tf.float32, shape = (None,), name = 'route_exists' )
    n_vertices  = tf.placeholder( tf.int32, shape = (None,), name = 'n_vertices')
    n_edges     = tf.placeholder( tf.int32, shape = (None,), name = 'edges')
    EV_matrix   = tf.placeholder( tf.float32, shape = (None,None), name = "EV" )
    edge_weight = tf.placeholder( tf.float32, shape = (None,1), name = "edge_weight" )
    target_cost = tf.placeholder( tf.float32, shape = (None,1), name = "target_cost" )
    time_steps  = tf.placeholder( tf.int32, shape = (), name = "time_steps" )
    
    
    # Compute initial embeddings for edges
    edge_initial_embeddings = graph_initiation_edge_MLP(tf.concat([ edge_weight, target_cost ], axis = 1))
    
    # All vertex embeddings are initialized with the same value, which is a trained parameter learned by the network
    totalNumberOfNodes = tf.shape(EV_matrix)[1]
    v_init = tf.get_variable(initializer=tf.random_normal((1,d)), dtype=tf.float32, name='V_init')
    vertex_initial_embeddings = tf.tile(
        tf.div(v_init, tf.sqrt(tf.cast(d, tf.float32))),
        [totalNumberOfNodes, 1]
    )

    # Define GNN dictionary
    GNN = {}

    # Define Graph neural network
    gnn = InstanceLoaderGraph(
        {
            # V is the set of vertex embeddings
            'V': d,
            'E': d
        },
        {
            'EV': ('E','V')
        },
        {
            'V_msg_E': ('V','E'),
            'E_msg_V': ('E','V')
        },
        {
            'V': [
                {
                    'mat': 'EV',
                    'msg': 'E_msg_V',
                    'transpose?': True,
                    'var': 'E'
                }
            ],
            'E': [
                {
                    'mat': 'EV',
                    'msg': 'V_msg_E',
                    'var': 'V'
                }
            ]
        },
        name='TSP'
    )

    # Populate GNN dictionary
    GNN['gnn']          = gnn
    GNN['route_exists'] = route_exists
    GNN['n_vertices']   = n_vertices
    GNN['n_edges']      = n_edges
    GNN['EV']           = EV_matrix
    GNN['W']            = edge_weight
    GNN['C']            = target_cost
    GNN['time_steps']   = time_steps

    # Define E_vote, which will compute one logit for each edge
    E_vote_MLP = Mlp(
        layer_sizes = [ d for _ in range(3) ],
        activations = [ tf.nn.relu for _ in range(3) ],
        output_size = 1,
        name = 'E_vote',
        name_internal_layers = True,
        kernel_initializer = tf.contrib.layers.xavier_initializer(),
        bias_initializer = tf.zeros_initializer()
        )
    
    # Get the last embeddings
    last_states = gnn(
      { "EV": EV_matrix, 'W': edge_weight, 'C': target_cost },
      { "V": vertex_initial_embeddings, "E": edge_initial_embeddings },
      time_steps = time_steps
    )
    GNN["last_states"] = last_states
    E_n = last_states['E'].h

    E_vote = tf.reshape(E_vote_MLP(E_n), [-1])

    # Compute the number of problems in the batch
    num_problems = tf.shape(n_vertices)[0]

    # Compute a logit probability for each problem
    pred_logits = tf.while_loop(
        lambda i, pred_logits: tf.less(i, num_problems),
        lambda i, pred_logits:
            (
                (i+1),
                pred_logits.write(
                    i,
                    tf.reduce_mean(E_vote[tf.reduce_sum(n_edges[0:i]):tf.reduce_sum(n_edges[0:i])+n_edges[i]])
                )
            ),
        [0, tf.TensorArray(size=num_problems, dtype=tf.float32)]
        )[1].stack()
    GNN['predictions'] = tf.sigmoid(pred_logits)

    GNN['TP'] = tf.reduce_sum(tf.multiply(route_exists, tf.cast(tf.equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['FP'] = tf.reduce_sum(tf.multiply(route_exists, tf.cast(tf.not_equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['TN'] = tf.reduce_sum(tf.multiply(tf.ones_like(route_exists)-route_exists, tf.cast(tf.equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['FN'] = tf.reduce_sum(tf.multiply(tf.ones_like(route_exists)-route_exists, tf.cast(tf.not_equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['acc'] = tf.reduce_mean(tf.cast(tf.equal(route_exists, tf.round(GNN['predictions'])), tf.float32))

    GNN['loss'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=route_exists, logits=pred_logits))

    optimizer = tf.train.Optimizer(name='Adam', learning_rate=learning_rate)

    vars_cost = tf.add_n([ tf.nn.l2_loss(var) for var in tf.trainable_variables() ])

    grads, _ = tf.clip_by_global_norm(tf.gradients(GNN['loss'] + tf.multiply(vars_cost, l2norm_scaling),tf.trainable_variables()),global_norm_gradient_clipping_ratio)
    GNN['train_step'] = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))
    
    return GNN
#end
