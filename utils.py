import glob
import io
import base64
from time import sleep
import numpy as np
import pandas as pd
import gym
from gym.wrappers import Monitor
from typing import Callable
from IPython.display import HTML
from IPython import display as ipythondisplay
from IPython.display import clear_output
import matplotlib.pyplot as plt


def show_video():
    mp4list = glob.glob("video/*.mp4")
    if len(mp4list) > 0:
        for video_name in mp4list:
            video = io.open(video_name, "r+b").read()
            encoded = base64.b64encode(video)
            ipythondisplay.display(
                HTML(
                    data="""<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>""".format(
                        encoded.decode("ascii")
                    )
                )
            )
    else:
        print("Could not find video")


def wrap_env(env):
    env = Monitor(env, "./video", force=True)
    return env
