import ai8x
from argparse import ArgumentParser
import torchinfo
import os
import fnmatch
from pydoc import locate
from devices import device

supported_models = []
model_names = []
for _, _, files in sorted(os.walk("models")):
    for name in sorted(files):
        if not fnmatch.fnmatch(name, "*.py"):
            continue
        fn = "models." + name[:-3]
        m = locate(fn)
        try:
            for i in m.models:
                i["module"] = fn
            supported_models += m.models
            model_names += [item["name"] for item in m.models]
        except AttributeError:
            # Skip files that don't have 'models' or 'models.name'
            pass


parser = ArgumentParser()
parser.add_argument("--device", type=device, default=85, help="Device name")
parser.add_argument(
    "--model", type=str, required=True, help="Model name", choices=model_names
)
parser.add_argument("--depth", "-d", type=int, default=1, help="Depth of summary")
parser.add_argument(
    "--col-names",
    type=str,
    nargs="*",
    default=["input_size", "output_size", "num_params", "params_percent"],
    help="Column names",
)
parser.add_argument(
    "--input-shape", type=int, nargs="+", required=True, help="Input shape"
)
parser.add_argument(
    "--kwargs",
    nargs="*",
    default=[],
    help="Additional key=value pairs to pass to Model (e.g. --kwargs dim=10 activation=relu)",
)

args = parser.parse_args()


def parse_kwargs(kwargs_str):
    def convert_value(value_str):
        # Handle tuples
        if value_str.startswith("(") and value_str.endswith(")"):
            # Remove parentheses and split by comma
            tuple_str = value_str[1:-1]
            # Handle empty tuple
            if not tuple_str:
                return tuple()
            # Split and convert each element
            elements = [convert_value(elem.strip()) for elem in tuple_str.split(",")]
            return tuple(elements)

        # Handle booleans
        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"

        # Handle integers
        try:
            return int(value_str)
        except ValueError:
            pass

        # Handle floats
        try:
            return float(value_str)
        except ValueError:
            pass

        # Return as string if no other type matches
        return value_str

    kwargs = dict()
    for item in kwargs_str:
        try:
            key, value = item.split("=")
            kwargs[key] = convert_value(value)
        except ValueError:
            print(f"Warning: Skipping malformed kwarg '{item}'")
    return kwargs


kwargs = parse_kwargs(args.kwargs)

ai8x.set_device(
    args.device,
    False,
    False,
    verbose=False,
)

module = supported_models[model_names.index(args.model)]
Model = locate(module["module"] + "." + args.model)
model = Model(**kwargs)

stats = torchinfo.summary(
    model,
    [1] + args.input_shape,
    device="cpu",
    depth=args.depth,
    col_names=args.col_names,
    mode="eval",
)

print(f"Float32 size: {(stats.total_params*4)/2**20:.2f} MB")
print(f"int8 size: {stats.total_params/2**20:.2f} MB")
