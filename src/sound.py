import multiprocessing
from playsound import playsound

class Sound:

    def __init__(self, sound):
        self.sound = sound

    def start(self):
        self.sound_process = multiprocessing.Process(target=playsound, args=("../audio/" + self.sound,))
        self.sound_process.start()

    def stop(self):
        self.sound_process.terminate()
