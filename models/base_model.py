
class BaseModel(object):
    def __init__(self):
        self.model = None
        self.inputs = []
        self.outputs = []

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model ({})...".format(checkpoint_path))
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def summary(self):
        if self.model is None:
            raise Exception("You have to build the model first.")

        return self.model.summary()

    def predict(self, x):
        if self.model is None:
            raise Exception("You have to build the model first.")

        return self.model.predict(x)

    def __call__(self, inputs, **kwargs):
        if self.model is None:
            raise Exception("You have to build the model first.")

        return self.model(inputs, **kwargs)

    def build_model(self):
        raise NotImplementedError
