import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello(){
    return "终于配好环境了，下班";
}
"""
my_extension = load_inline(name="my_extension", cpp_sources=[cpp_source], functions=["hello"], verbose=True)

print(my_extension.hello())
