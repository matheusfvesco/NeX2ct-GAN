import logging
from logging.handlers import RotatingFileHandler
import datetime
from pathlib import Path
import tqdm
import zoneinfo


class TqdmLoggingHandler(logging.Handler):
    """
    Passes console outputs to tqdm write instead
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(
    logger_name: str = None,
    experiment_name: str = "",
    experiment_description: str = "",
    filename_comment: str = "",
    log_dir: str | Path = "logs",
    log_file: str = "log_{now}.log",
    strftime: str = "%Y-%m-%d_%H-%M",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    *args,
    **kwargs,
):
    header = """
    Experiment Details:
    - Name: {experiment_name}
    - Date: {now}
    - Description: {experiment_description}"""

    # ensuring timezone is correct
    tzinfo = zoneinfo.ZoneInfo("America/Sao_Paulo")
    now = datetime.datetime.now(tzinfo).strftime(strftime)
    logging.Formatter.converter = lambda *args: datetime.datetime.now(
        tzinfo
    ).timetuple()

    # updates kwargs
    kwargs.update(
        {
            "now": now,
            "experiment_name": experiment_name or logger_name,
            "experiment_description": experiment_description,
        }
    )

    # defaults logger to root
    logger_name = logger_name if logger_name is not None else "root"

    # if filename_comment, adds it to the end of the filename
    log_file = (
        log_file
        if not filename_comment
        else log_file.replace(".log", f"_{filename_comment}.log")
    )

    def format_string(string: str):
        for keyword, value in kwargs.items():
            string = string.replace(("{" + keyword + "}"), value)
        return string

    # ensures parent directories exist
    log_file = format_string(log_file)
    log_file: Path = Path(log_dir) / logger_name / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    console_handler = TqdmLoggingHandler()
    file_handler = RotatingFileHandler(log_file)
    console_handler.setLevel(console_level)
    file_handler.setLevel(file_level)

    console_format = logging.Formatter("%(levelname)s: %(name)s - %(message)s")
    file_format = logging.Formatter(
        "%(levelname)s: %(asctime)s\n%(filename)s:%(lineno)d - %(name)s - %(message)s\n"
    )

    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging {logger_name}\n")

    info_header = "\n\n" + "=" * 40 + format_string(header) + "\n" + "=" * 40 + "\n"
    logger.info(info_header)
    # logger.info(f"\n{format_string(header)}\n")

    return logger
