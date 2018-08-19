class Dataset:
    def __init__(self, train, test, values, labels, batch_size):
        self.train = train.repeat().batch(batch_size)
        self.test = test.repeat().batch(batch_size)
        self.values = values
        self.labels = labels
        input_shape, label_shape = train.output_shapes
        assert train.output_shapes == test.output_shapes
        assert len(input_shape) == 3
        self.height, self.width, self.channels = input_shape.as_list()