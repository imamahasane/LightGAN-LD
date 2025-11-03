from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def make_tb_logger(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir)
