import time


class Profile:
    def __init__(self):
        self.start = 0.0
        self.data = []

    def begin(self):
        self.start = time.time()
        del self.data[:]

    def print_time(self):
        print 'Time elapsed: ', self.get_time()

    def get_time(self):
        return time.time() - self.start

    def print_data(self):
        print self.data

    def get_data(self):
        return self.data

    def save(self, msg=' '):
        self.data.append([self.get_time(), msg])
