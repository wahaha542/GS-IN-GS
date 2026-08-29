import os
import sys
import importlib.util

# 先导入torch
import torch

# 添加必要的DLL路径到系统PATH
# CUDA 11.8 DLL路径
cuda_bin_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin'
if cuda_bin_path not in os.environ['PATH']:
    os.environ['PATH'] = cuda_bin_path + ';' + os.environ['PATH']

# PyTorch CUDA DLL路径
torch_cuda_path = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), 'lib')
if os.path.exists(torch_cuda_path) and torch_cuda_path not in os.environ['PATH']:
    os.environ['PATH'] = torch_cuda_path + ';' + os.environ['PATH']

# 调试信息
print(f"Python version: {sys.version}")
print(f"Looking for simple_knn._C module")
print(f"Current directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")
print(f"CUDA bin path added: {cuda_bin_path}")
if torch_cuda_path:
    print(f"PyTorch CUDA path added: {torch_cuda_path}")
print(f"System PATH: {os.environ['PATH'][:500]}...")

# 检查所有可用的_C模块
module_dir = os.path.dirname(__file__)
print(f"Module directory: {module_dir}")
_c_files = [f for f in os.listdir(module_dir) if f.startswith('_C.') and f.endswith('.pyd')]
print(f"Available _C modules: {_c_files}")

# 尝试加载合适的模块
try:
    # 首先尝试直接导入
    from simple_knn._C import distCUDA2
    print("Successfully imported distCUDA2 from simple_knn._C")
except ImportError as e:
    print(f"Direct import failed: {e}")
    
    # 尝试直接加载.pyd文件
    for c_file in _c_files:
        try:
            c_path = os.path.join(module_dir, c_file)
            spec = importlib.util.spec_from_file_location("simple_knn._C", c_path)
            if spec and spec.loader:
                _C = importlib.util.module_from_spec(spec)
                sys.modules["simple_knn._C"] = _C
                spec.loader.exec_module(_C)
                distCUDA2 = _C.distCUDA2
                print(f"Successfully loaded {c_file} and imported distCUDA2")
                break
        except Exception as e2:
            print(f"Failed to load {c_file}: {e2}")
    else:
        raise ImportError("Could not load any simple_knn._C module") from e
