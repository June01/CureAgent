"""
GPU 兼容性工具模块
支持 NVIDIA CUDA 和 AMD ROCm GPU
"""

import torch
import gc
import os
import logging

logger = logging.getLogger(__name__)


class GPUManager:
    """GPU 管理器，自动检测和管理 CUDA/ROCm GPU"""
    
    def __init__(self):
        self.device_type = self._detect_device_type()
        self.device_count = self._get_device_count()
        self.current_device = self._get_current_device()
        
        logger.info(f"检测到 GPU 类型: {self.device_type}")
        logger.info(f"可用设备数量: {self.device_count}")
        
    def _detect_device_type(self):
        """检测GPU类型"""
        if torch.cuda.is_available():
            # 检查是否为 AMD GPU (ROCm)
            try:
                device_name = torch.cuda.get_device_name(0).lower()
                if 'amd' in device_name or 'radeon' in device_name or 'gfx' in device_name:
                    return 'rocm'
                else:
                    return 'cuda'
            except:
                return 'cuda'  # 默认假设为 CUDA
        else:
            return 'cpu'
    
    def _get_device_count(self):
        """获取设备数量"""
        if self.device_type in ['cuda', 'rocm']:
            return torch.cuda.device_count()
        return 0
    
    def _get_current_device(self):
        """获取当前设备"""
        if self.device_type in ['cuda', 'rocm']:
            return torch.cuda.current_device()
        return None
    
    def empty_cache(self):
        """清空GPU缓存"""
        try:
            if self.device_type in ['cuda', 'rocm']:
                torch.cuda.empty_cache()
                logger.debug(f"已清空 {self.device_type.upper()} 缓存")
            gc.collect()
        except Exception as e:
            logger.warning(f"清空GPU缓存时出错: {e}")
    
    def get_memory_info(self):
        """获取GPU内存信息"""
        if self.device_type in ['cuda', 'rocm']:
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                return {
                    'allocated': allocated,
                    'reserved': reserved,
                    'allocated_gb': allocated / 1024**3,
                    'reserved_gb': reserved / 1024**3
                }
            except Exception as e:
                logger.warning(f"获取GPU内存信息时出错: {e}")
                return None
        return None
    
    def set_device(self, device_id=0):
        """设置当前设备"""
        if self.device_type in ['cuda', 'rocm'] and device_id < self.device_count:
            torch.cuda.set_device(device_id)
            self.current_device = device_id
            logger.info(f"切换到设备 {device_id}")
        else:
            logger.warning(f"无法切换到设备 {device_id}")
    
    def get_device_name(self, device_id=0):
        """获取设备名称"""
        if self.device_type in ['cuda', 'rocm'] and device_id < self.device_count:
            try:
                return torch.cuda.get_device_name(device_id)
            except Exception as e:
                logger.warning(f"获取设备名称时出错: {e}")
                return f"Unknown {self.device_type.upper()} Device"
        return "CPU"
    
    def is_available(self):
        """检查GPU是否可用"""
        return self.device_type in ['cuda', 'rocm'] and torch.cuda.is_available()
    
    def get_device_properties(self, device_id=0):
        """获取设备属性"""
        if self.device_type in ['cuda', 'rocm'] and device_id < self.device_count:
            try:
                props = torch.cuda.get_device_properties(device_id)
                return {
                    'name': props.name,
                    'major': props.major,
                    'minor': props.minor,
                    'total_memory': props.total_memory,
                    'total_memory_gb': props.total_memory / 1024**3,
                    'multi_processor_count': props.multi_processor_count
                }
            except Exception as e:
                logger.warning(f"获取设备属性时出错: {e}")
                return None
        return None
    
    def optimize_for_inference(self):
        """为推理优化GPU设置"""
        if self.device_type == 'rocm':
            # AMD ROCm 特定优化
            os.environ.setdefault('HSA_FORCE_FINE_GRAIN_PCIE', '1')
            os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
            logger.info("应用 ROCm 推理优化设置")
        elif self.device_type == 'cuda':
            # NVIDIA CUDA 特定优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("应用 CUDA 推理优化设置")
    
    def print_info(self):
        """打印GPU信息"""
        print(f"🔧 GPU 管理器信息:")
        print(f"   设备类型: {self.device_type.upper()}")
        print(f"   设备数量: {self.device_count}")
        print(f"   当前设备: {self.current_device}")
        
        if self.is_available():
            for i in range(self.device_count):
                name = self.get_device_name(i)
                props = self.get_device_properties(i)
                print(f"   设备 {i}: {name}")
                if props:
                    print(f"     总内存: {props['total_memory_gb']:.1f} GB")
                    print(f"     计算能力: {props['major']}.{props['minor']}")
        else:
            print("   ⚠️  没有可用的GPU设备")


# 全局GPU管理器实例
gpu_manager = GPUManager()


def empty_cache():
    """清空GPU缓存的便捷函数"""
    gpu_manager.empty_cache()


def get_device():
    """获取推荐的设备"""
    if gpu_manager.is_available():
        return f"cuda:{gpu_manager.current_device}"
    return "cpu"


def is_gpu_available():
    """检查GPU是否可用"""
    return gpu_manager.is_available()


def optimize_for_inference():
    """为推理优化GPU设置"""
    gpu_manager.optimize_for_inference()


def print_gpu_info():
    """打印GPU信息"""
    gpu_manager.print_info()


# 自动优化设置
if gpu_manager.is_available():
    optimize_for_inference()
