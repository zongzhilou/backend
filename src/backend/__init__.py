"""昇腾 NPU 平台实现。

AscendPlatform 基于 torch_npu 提供设备管理、随机数、显存、集合通信、
Ray 资源与性能分析能力，只依赖 torch / torch_npu / packaging，可独立
实例化。文件末尾的受保护块把该实现接入 verl 与 Megatron-LM：verl
存在时注册为 ascend / huawei 平台；Megatron-LM 与 megatron_adaptor
同时存在时激活其昇腾适配。宿主缺失时对应块自动跳过。
"""

from __future__ import annotations

import logging
import os
import platform as _platform
import re
import shutil
import subprocess
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Optional

# torch 2.12+ 的设备后端自动加载会在 import torch 时尝试初始化 torch_npu，
# CANN 缺失的进程会直接失败。本插件自行管理 torch_npu 的导入
# （见 _ensure_torch_npu），无需自动加载；setdefault 允许外部显式开启。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from packaging.version import parse as _version_parse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["AscendPlatform", "main"]


def _ensure_torch_npu() -> bool:
    """尝试导入 torch_npu；返回导入后 torch.npu 是否可用。"""
    if hasattr(torch, "npu"):
        return True
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu")
    except Exception as e:
        logger.debug("The current machine has no torch.npu, because: %s", e)
    return False


_ensure_torch_npu()  # 模块加载时预导入，加速后续可用性判断。

# npu-smi 的常见安装路径。
_NPU_SMI_CANDIDATES = (
    "npu-smi",
    "/usr/local/Ascend/driver/tools/npu-smi",
    "/usr/local/Ascend/driver/bin/npu-smi",
    "/usr/local/Ascend/ascend-toolkit/latest/bin/npu-smi",
)


def _npu_smi_available() -> bool:
    """探测昇腾驱动：任一候选路径的 ``npu-smi info`` 退出码为 0。"""
    for cmd in _NPU_SMI_CANDIDATES:
        cmd_path = shutil.which(cmd)
        if cmd_path is None:
            continue
        try:
            result = subprocess.run(
                [cmd_path, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode == 0:
            return True
    return False


class AscendHardwareVersion(IntEnum):
    """NPU 芯片代际。"""

    NONE = 0
    A2 = 2
    A3 = 3
    A5 = 5
    MAX_VERSION = 999


def get_npu_versions() -> tuple[AscendHardwareVersion, str, str]:
    """返回 (芯片代际, 软件版本, CANN 版本)。

    软件版本经 ``npu-smi info -t board`` 解析：卡 1 不存在时（K8s 常见的
    非连续卡号）改用 ASCEND_VISIBLE_DEVICES 的首个可见卡；A3 一卡双 die
    的 die 序号需换算为物理卡号。CANN 版本读 ascend_toolkit_install.info。

    Raises:
        RuntimeError: 无法取得任一版本信息。
    """
    try:
        import torch_npu

        device_name = torch_npu.npu.get_device_name()
    except Exception:
        hardware_version = AscendHardwareVersion.NONE
    else:
        if "Ascend910_95" in device_name or "Ascend950" in device_name:
            hardware_version = AscendHardwareVersion.A5
        elif "Ascend910_93" in device_name:
            hardware_version = AscendHardwareVersion.A3
        elif "Ascend910B" in device_name or "A2G" in device_name:
            hardware_version = AscendHardwareVersion.A2
        else:
            hardware_version = AscendHardwareVersion.MAX_VERSION

    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "board", "-i", "1"], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError:
        visible_devices = os.environ.get("ASCEND_VISIBLE_DEVICES")
        if not visible_devices:
            raise

        try:
            npu_id = int(visible_devices.split(",")[0])
        except (ValueError, IndexError):
            raise

        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "board", "-i", str(npu_id)], capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError:
            physical_card_id = npu_id // 2
            result = subprocess.run(
                ["npu-smi", "info", "-t", "board", "-i", str(physical_card_id)],
                capture_output=True,
                text=True,
                check=True,
            )

    software_version = None
    for line in result.stdout.split("\n"):
        if "Software Version" in line:
            parts = line.split(":")
            if len(parts) > 1:
                software_version = parts[1].strip().lower()
            break

    if not software_version:
        raise RuntimeError("Could not find Software Version in npu-smi output")

    arch = _platform.machine()
    if arch not in ["arm64", "aarch64", "x86_64"]:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    cann_path = os.path.join(ascend_home, f"{arch}-linux")
    info_file = os.path.join(cann_path, "ascend_toolkit_install.info")
    if not os.path.exists(info_file):
        raise RuntimeError(f"CANN toolkit info file does not exist: {info_file}")

    cann_version = None
    with open(info_file) as f:
        for line in f:
            if line.startswith("version="):
                cann_version = line.split("=", 1)[1].strip().lower()
                break

    if not cann_version:
        raise RuntimeError("Could not find version in CANN toolkit info file")

    return hardware_version, software_version, cann_version


def check_ipc_version_support(software_version: str, cann_version: str) -> bool:
    """判定 IPC 支持：software >= 25.3.rc1 且 CANN >= 8.3.rc1。

    版本串兼容 rc 后缀（如 25.3.rc1.2）与小写 t 修订（如 25.5.t3.b001）。

    Raises:
        RuntimeError: 版本号格式无法解析。
    """
    # 含小写 t 的版本只取前两段，其余取至三段。
    ascend_version_pattern = r"(\d+\.\d+(?=\.t))|(\d+\.\d+(?:\.(?:rc\d+|\d+))?)"
    software_match = re.match(ascend_version_pattern, software_version)
    if not software_match:
        raise RuntimeError(f"Invalid software version format: {software_version}")

    software_base = software_match.group(1) if software_match.group(1) else software_match.group(2)

    cann_match = re.match(ascend_version_pattern, cann_version)
    if not cann_match:
        raise RuntimeError(f"Invalid CANN version format: {cann_version}")
    else:
        cann_base = cann_match.group(1) if cann_match.group(1) else cann_match.group(2)

    if _version_parse(software_base) >= _version_parse("25.3.rc1"):
        if _version_parse(cann_base) >= _version_parse("8.3.rc1"):
            return True
        else:
            logger.info(f"CANN version {cann_version} is below 8.3.RC1")
    else:
        logger.info(f"Software version {software_version} is below 25.3.rc1")

    return False


class AscendPlatform:
    """昇腾 NPU 平台核心实现。

    方法面遵循平台抽象的鸭子类型约定（device_name / is_available /
    manual_seed / Ray 资源描述等），不继承任何宿主基类，可独立实例化；
    宿主适配见 ``_register_verl_platform``。
    """

    @staticmethod
    def check_smi_command(cmd: str) -> bool:
        """执行 SMI 命令，退出码为 0 返回 True。"""
        cmd_path = shutil.which(cmd)
        if cmd_path is None:
            return False
        try:
            result = subprocess.run([cmd_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    @property
    def device_name(self) -> str:
        return "npu"

    @property
    def vendor_name(self) -> str:
        return "ascend"

    @property
    def device_module(self):
        return torch.npu

    def is_available(self) -> bool:
        return torch.npu.is_available()

    def is_platform_available(self, use_smi_check: bool = False) -> bool:
        """返回本机是否为可用昇腾环境。

        ``use_smi_check=True`` 供宿主自动探测：torch_npu 可导入即认定是
        昇腾环境；再以 npu-smi 兜底，覆盖 torch_npu 未安装的进程
        （CPU-only Ray actor / 极简容器）。
        """
        if not _ensure_torch_npu():
            return bool(use_smi_check and _npu_smi_available())
        if use_smi_check:
            return True
        return torch.npu.is_available()

    def current_device(self) -> int:
        return torch.npu.current_device()

    def device_count(self) -> int:
        return torch.npu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.npu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.npu.synchronize(device_index)

    def manual_seed(self, seed: int) -> None:
        torch.npu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.npu.manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        try:
            torch.npu.memory._set_allocator_settings(settings)
        except Exception:
            logger.warning(
                "Current version of torch-npu does not support `_set_allocator_settings`, "
                "please upgrade torch-npu to 2.9.0 or later"
            )

    def empty_cache(self) -> None:
        torch.npu.empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # torch_npu 未安装的进程（如 CPU-only Ray actor）没有 torch.npu，
        # 直接访问会 AttributeError；按契约返回 (None, None)。
        if not hasattr(torch, "npu"):
            return (None, None)
        if not hasattr(torch.npu, "get_device_capability"):
            return (None, None)
        result = torch.npu.get_device_capability(device_index)
        if result is None:
            return (None, None)
        return result

    def communication_backend_name(self) -> str:
        return "flagcx" if os.getenv("USE_FLAGCX", "0").lower() in ["1", "true"] else "hccl"

    def visible_devices_envvar(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    def ray_resource_name(self) -> str:
        return "NPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"resources": {"NPU": num_gpus}}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"]

    def get_device_uuid(self, device_id) -> str:
        visible_devices = os.environ.get(self.visible_devices_envvar())
        if visible_devices is not None:
            visible_list = visible_devices.split(",")
            assert device_id < len(visible_list), f"device_id {device_id} must less than {len(visible_list)}"
            return self.ray_resource_name() + visible_list[device_id]
        return f"{self.ray_resource_name()}-{device_id}"

    def rollout_env_vars(self) -> dict[str, str]:
        # 防止 disaggregated 模式下 actor 与 rollout 权重同步挂起或崩溃。
        env_vars = {"NCCL_CUMEM_ENABLE": "0", "VLLM_ASCEND_AUTO_DETECT_QUANTIZATION": "0"}
        if os.environ.get("VLLM_ASCEND_TASK_QUEUE_ENABLE", None):
            env_vars["TASK_QUEUE_ENABLE"] = os.environ["VLLM_ASCEND_TASK_QUEUE_ENABLE"]
        return env_vars

    def is_ipc_supported(self) -> bool:
        """A5 系直接支持；其余按软件/CANN 版本判定。"""
        try:
            hardware_version, software_version, cann_version = get_npu_versions()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute npu-smi command: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error checking IPC support: {e}") from e

        if hardware_version == AscendHardwareVersion.A5:
            return True
        return check_ipc_version_support(software_version, cann_version)

    @contextmanager
    def nvtx_range(self, msg: str):
        """mstx 是昇腾的 NVTX 等价物；不可用时退化为 debug 日志。"""
        mstx = getattr(getattr(torch, "npu", None), "mstx", None)
        if mstx is not None and hasattr(mstx, "mstx_range"):
            with mstx.mstx_range(msg):
                yield
        else:
            logger.debug("NVTX range (mstx unavailable, no-op on NPU): %s", msg)
            yield

    def profiler_start(self) -> None:
        try:
            import torch_npu
        except ImportError:
            logger.debug("torch_npu unavailable; profiler start skipped")
            return
        if hasattr(torch_npu, "profiler") and hasattr(torch_npu.profiler, "start"):
            torch_npu.profiler.start()
        else:
            logger.debug("torch_npu.profiler.start unavailable; skipped")

    def profiler_stop(self) -> None:
        try:
            import torch_npu
        except ImportError:
            logger.debug("torch_npu unavailable; profiler stop skipped")
            return
        if hasattr(torch_npu, "profiler") and hasattr(torch_npu.profiler, "stop"):
            torch_npu.profiler.stop()
        else:
            logger.debug("torch_npu.profiler.stop unavailable; skipped")

    def cudart(self) -> Any:
        return None


# ---------------------------------------------------------------------------
# 宿主适配（受保护导入：宿主缺失时自动跳过）
# ---------------------------------------------------------------------------


def _register_verl_platform() -> None:
    """把核心平台注册进 verl 的 PlatformRegistry；verl 不存在时跳过。

    verl 的 ``PlatformRegistry.register`` 强制 ``issubclass(cls, PlatformBase)``，
    因此适配子类只能在确认 verl 可导入后创建。继承顺序把核心实现放在
    MRO 前部：PlatformBase 的抽象方法全部落地，且核心实现优先于其默认
    实现。"huawei" 是 verl 内置 NPU 平台的既有注册名，与 "ascend" 同时
    注册以保证 VERL_PLATFORM=ascend 与主机自动探测行为一致。
    """
    try:
        from verl.plugin.platform import PlatformBase, PlatformRegistry
    except ImportError:
        logger.debug("verl 不可用，跳过 verl 平台注册")
        return

    @PlatformRegistry.register(platform="ascend")
    @PlatformRegistry.register(platform="huawei")
    class PlatformAscend(AscendPlatform, PlatformBase):
        """verl PlatformBase 契约下的昇腾平台。"""

    logger.debug("已向 verl 注册平台: ascend / huawei")


def _activate_megatron_adaptor() -> None:
    """激活 Megatron-LM 的昇腾适配；megatron 或 megatron_adaptor 缺一即跳过。

    等价于在训练脚本中执行 ``import megatron_adaptor``：其内部对
    Megatron-LM 做 monkey-patch，并自动加载 mindspeed（如已安装）。
    """
    import importlib.util

    if importlib.util.find_spec("megatron") is None:
        logger.debug("megatron 未安装，跳过 Megatron-LM 昇腾适配")
        return
    if importlib.util.find_spec("megatron_adaptor") is None:
        logger.debug("megatron_adaptor 未安装，跳过 Megatron-LM 昇腾适配")
        return
    try:
        import megatron_adaptor  # noqa: F401

        logger.debug("megatron_adaptor 昇腾适配已激活")
    except Exception as e:
        logger.warning("激活 megatron_adaptor 失败: %s", e)


_register_verl_platform()
_activate_megatron_adaptor()


def main() -> None:
    """``backend`` 命令行：打印核心平台状态与 verl 注册表（如已安装）。"""
    core = AscendPlatform()
    print(f"backend 核心平台: {core.vendor_name} / {core.device_name}")
    print(f"torch_npu 可导入: {hasattr(torch, 'npu')}，NPU 可用: {core.is_available() if hasattr(torch, 'npu') else False}")
    print(f"驱动 npu-smi 在位: {_npu_smi_available()}")
    try:
        from verl.plugin.platform import PlatformRegistry
    except ImportError:
        print("verl 未安装，跳过 verl 注册表查询")
        return
    print(f"verl 已注册平台: {', '.join(PlatformRegistry.registered_names())}")
