from mutransformers import BertConfig, BertForMaskedLM
from mup import make_base_shapes, set_base_shapes
from functools import partial
# define a base model
base_config = BertConfig(
    hidden_size=256,
    intermediate_size=256,
    num_attention_heads=16,
)
base_model = BertForMaskedLM(config=base_config)
# define a delta models where we vary all "widths" we want to vary
delta_config = BertConfig(
    hidden_size=200,
    intermediate_size=300,
    num_attention_heads=5,
)
delta_model = BertForMaskedLM(config=delta_config)
# define a base shape object based on comparing delta_model against base_model
base_shapes = make_base_shapes(base_model, delta_model, savefile='bert256.bsh')
print(base_shapes)
# define target model
target_config = BertConfig(
    hidden_size=1024,
    intermediate_size=1024*4,
    num_attention_heads=32,
)
target_model = BertForMaskedLM(config=target_config)

# set base shapes
set_base_shapes(target_model, base_shapes)
# you can alternatively load base shape from file
# set_base_shapes(target_model, 'bert256.bsh')

# re-initialize
# target_model.apply(target_model._init_weights)

# train target_model, etc
