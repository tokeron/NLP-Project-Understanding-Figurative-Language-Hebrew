import yaml
import sys

f = open("config.yaml")
y = yaml.safe_load(f)
y["data_args"]["dataset"] = sys.argv[1]
y["model_args"]["learning_rate"] = float(sys.argv[2])
y["model_args"]["num_epochs"] = int(sys.argv[3])
y["data_args"]["ignore_subtokens"] = bool(int(sys.argv[4]))
f = open("config.yaml", "w")
yaml.dump(y, f)
f.close()

