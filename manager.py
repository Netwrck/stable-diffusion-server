# poll the progress.txt file forever
import os
from datetime import datetime
from time import sleep

from loguru import logger

while True:
    try:
        with open("progress.txt", "r") as f:
            progress = f.read()
            last_mod_time = datetime.fromtimestamp(os.path.getmtime("progress.txt"))
            if (datetime.now() - last_mod_time).seconds > 60 * 7:
                # no progress for 7 minutes, restart/kill with -9
                logger.info("restarting server to fix cuda issues (device side asserts)")
                os.system("/usr/bin/bash kill -SIGHUP `pgrep gunicorn`")
                os.system("/usr/bin/bash kill -SIGHUP `pgrep uvicorn`")
                os.system("kill -9 `pgrep gunicorn`")
                os.system("kill -9 `pgrep uvicorn`")
                os.system("killall -9 uvicorn")
                os.system("ps | grep uvicorn | awk '{print $1}' | xargs kill -9")

            if progress == "done":
                break
    except Exception as e:
        print(e)
        pass
    sleep(60*5)
