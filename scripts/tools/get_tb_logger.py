import os

from lib.utils.tools.logger import Logger as Log
from tensorboardX import SummaryWriter
from datetime import datetime


def get_tb_logger(configer):
    tb_logger = None
    if configer.exists("logging","tb_logger"):
        if not os.path.exists(configer.get("logging","tb_logger")):
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = configer.get("logging","tb_name")
            tb_logger = SummaryWriter(
                os.path.join(configer.get("logging","tb_logger"), "log/" + save_name + "_" + current_time)
            )
    return tb_logger