import datetime
import os

class Logger:
    def __init__(self, exp_name):
        if not os.path.exists("logs"):
            os.mkdir("logs")
        self.file = open('logs/{}.log'.format(exp_name), 'w')

    def log(self, content):
        self.file.write(content + '\n')
        self.file.flush()



