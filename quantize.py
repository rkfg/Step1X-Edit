import sys

from safetensors.torch import load_file, save_file
from torch import float8_e4m3fn, float8_e5m2

if len(sys.argv) != 4:
    print("Provide input/output file names and fp8 type as either e4m3 or e5m2")
    exit(1)

if sys.argv[3] == "e4m3":
    dt = float8_e4m3fn
elif sys.argv[3] == "e5m2":
    dt = float8_e5m2
else:
    print("Invalid quantization type, should be either e4m3 or e5m2")
    exit(1)

state_dict = load_file(sys.argv[1])

for k in state_dict:
    if "norm" in k or "bias" in "k":
        state_dict[k] = state_dict[k].bfloat16()
    else:
        state_dict[k] = state_dict[k].to(dt)

save_file(state_dict, sys.argv[2])
