TENSORNET_QM9_PARM = {"precision":32,"embedding_dimension":256,"num_layers":8,"num_rbf":64,"rbf_type":"expnorm",
                  "trainable_rbf":False,"activation":"silu","cutoff_lower":0.0,"cutoff_upper":5.0,"max_z":100,
                  "max_num_neighbors":64,"model":"tensornet","aggr":"add","neighbor_embedding":True,
                  "attn_activation":"silu","num_heads":8,"distance_influence":"both",
                  "equivariance_invariance_group":"O(3)","atom_filter":-1,"prior_model":None,"output_model":"Scalar",
                  "reduce_op":"add","derivative":True}
ET_QM9_PARM = {"precision":32,"embedding_dimension":256,"num_layers":8,"num_rbf":64,"rbf_type":"expnorm",
              "trainable_rbf":False,"activation":"silu","cutoff_lower":0.0,"cutoff_upper":5.0,"max_z":100,
              "max_num_neighbors":64,"model":"equivariant-transformer","aggr":"add","neighbor_embedding":True,
              "attn_activation":"silu","num_heads":8,"distance_influence":"both",
              "equivariance_invariance_group":"O(3)","atom_filter":-1,"prior_model":None,"output_model":"Scalar",
              "reduce_op":"add","derivative":True}
TENSORNET_QM9_NO_F_PARM = {"precision":32,"embedding_dimension":256,"num_layers":8,"num_rbf":64,"rbf_type":"expnorm",
                  "trainable_rbf":False,"activation":"silu","cutoff_lower":0.0,"cutoff_upper":5.0,"max_z":100,
                  "max_num_neighbors":64,"model":"tensornet","aggr":"add","neighbor_embedding":True,
                  "attn_activation":"silu","num_heads":8,"distance_influence":"both",
                  "equivariance_invariance_group":"O(3)","atom_filter":-1,"prior_model":None,"output_model":"Scalar",
                  "reduce_op":"add","derivative":False}
ET_QM9_NO_F_PARM = {"precision":32,"embedding_dimension":256,"num_layers":8,"num_rbf":64,"rbf_type":"expnorm",
              "trainable_rbf":False,"activation":"silu","cutoff_lower":0.0,"cutoff_upper":5.0,"max_z":100,
              "max_num_neighbors":64,"model":"equivariant-transformer","aggr":"add","neighbor_embedding":True,
              "attn_activation":"silu","num_heads":8,"distance_influence":"both",
              "equivariance_invariance_group":"O(3)","atom_filter":-1,"prior_model":None,"output_model":"Scalar",
              "reduce_op":"add","derivative":False}

ET_MD17_PARM = {"precision":32,"embedding_dimension":128,"num_layers":6,"num_rbf":32,"rbf_type":"expnorm",
              "trainable_rbf":False,"activation":"silu","cutoff_lower":0.0,"cutoff_upper":5.0,"max_z":100,
              "max_num_neighbors":64,"model":"equivariant-transformer","aggr":"add","neighbor_embedding":True,
              "attn_activation":"silu","num_heads":8,"distance_influence":"both",
              "equivariance_invariance_group":"O(3)","atom_filter":-1,"prior_model":None,"output_model":"Scalar",
              "reduce_op":"add","derivative":True}

ET_MD17_NO_F_PARM = {"precision":32,"embedding_dimension":128,"num_layers":6,"num_rbf":32,"rbf_type":"expnorm",
              "trainable_rbf":False,"activation":"silu","cutoff_lower":0.0,"cutoff_upper":5.0,"max_z":100,
              "max_num_neighbors":64,"model":"equivariant-transformer","aggr":"add","neighbor_embedding":True,
              "attn_activation":"silu","num_heads":8,"distance_influence":"both",
              "equivariance_invariance_group":"O(3)","atom_filter":-1,"prior_model":None,"output_model":"Scalar",
              "reduce_op":"add","derivative":False}

TENSORNET_MD17_PARM = {"precision":32,"embedding_dimension":128,"num_layers":2,"num_rbf":32,"rbf_type":"expnorm",
              "trainable_rbf":False,"activation":"silu","cutoff_lower":0.0,"cutoff_upper":4.5,"max_z":128,
              "max_num_neighbors":64,"model":"tensornet","aggr":"add","neighbor_embedding":True,
              "attn_activation":"silu","num_heads":8,"distance_influence":"both",
              "equivariance_invariance_group":"O(3)","atom_filter":-1,"prior_model":None,"output_model":"Scalar",
              "reduce_op":"add","derivative":True}

TENSORNET_MD17_NO_F_PARM = {"precision":32,"embedding_dimension":128,"num_layers":2,"num_rbf":32,"rbf_type":"expnorm",
              "trainable_rbf":False,"activation":"silu","cutoff_lower":0.0,"cutoff_upper":4.5,"max_z":128,
              "max_num_neighbors":64,"model":"tensornet","aggr":"add","neighbor_embedding":True,
              "attn_activation":"silu","num_heads":8,"distance_influence":"both",
              "equivariance_invariance_group":"O(3)","atom_filter":-1,"prior_model":None,"output_model":"Scalar",
              "reduce_op":"add","derivative":False}