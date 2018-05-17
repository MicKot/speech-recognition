class Batcher():

    def __init__(self, signals, labels):
        self.i = 0
        self.signals = signals
        self.labels = labels

    def next_batch(self, batch_size):
        x = self.signals[self.i:self.i + batch_size]
        y = self.labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.signals)
        return x, y

    def next_batch_test(self, batch_size):
        x = self.signals[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.signals)
        return x