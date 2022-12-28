# parser --------------------------------------------------------------------------
import yaml
from box import Box

with open('/home/tok/figurative-language/config//config.yaml') as f:
    global training_args
    training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
