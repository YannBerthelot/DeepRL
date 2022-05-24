import argparse
import configparser
import platform
import torch


def read_config(config_file: str) -> configparser.ConfigParser:
    if not config_file.endswith(".ini"):
        raise ValueError(
            f'Configuration file {config_file} is in the {config_file.split(".")[-1]} extension, should be .ini'
        )
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
    except:
        raise OSError(f"Config file {config_file} is impossible to read")
    if torch.cuda.is_available():
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        GPU_NAME = "No GPU"
    try:
        CPU_NAME = platform.processor()
    except:
        print("Couldn't get processor name, using generic name instead")
        CPU_NAME = "CPU"
    config["HARDWARE"]["GPU_name"] = GPU_NAME
    config["HARDWARE"]["CPU_name"] = CPU_NAME
    return config


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-s")
    args = parse.parse_args()
    config = read_config(args.s)
    print(config.sections())
