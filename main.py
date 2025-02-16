from src.set_up_config_device import (
    get_allowed_cpu_count,
    set_up_config_device,
    set_up_device,
)
from src.setup_logger import setup_logger

# Initialize logger
logger = setup_logger()

device = set_up_device()

cpu_count = get_allowed_cpu_count()

n_process = set_up_config_device(cpu_count)
