import os
import importlib.util
import inspect


file_path = "evaluation/cross_validation.py"
module_name = os.path.splitext(os.path.basename(file_path))[0]
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Iterate through the module's members and print the function signatures
for name, obj in inspect.getmembers(module):
    if inspect.isfunction(obj):
        signature = inspect.signature(obj)
        print(f"Function: {name}\nSignature: {signature}\n")
