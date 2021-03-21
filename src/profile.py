from sound import Sound

class Profile:

    def __init__(self, nr_of_people, preferences):
        self.index = 0
        self.nr_of_people = nr_of_people
        self.playlist = []
        self.preferences = preferences
        for p in preferences:
            self.playlist.append(Sound(p))

    def next_song(self):
        if len(self.playlist) > 0:
            self.playlist[self.index].stop()
            self.index = (self.index + 1) % len(self.playlist)
            self.playlist[self.index].start()

    def start(self):
        if len(self.playlist) > 0:
            self.playlist[self.index].start()

    def stop(self):
        if len(self.playlist) > 0:
            self.playlist[self.index].stop()

    # get song that is playing in string format
    def get_current_song(self):
        if len(self.playlist) > 0:
            return self.preferences[self.index]
        else:
            return ""
