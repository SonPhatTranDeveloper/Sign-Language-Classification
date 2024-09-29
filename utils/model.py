import pickle


class ASLClassificationModel:
    @staticmethod
    def load_model(model_path):
        # Load model and mapping from pickle
        with open(model_path, "rb") as file:
            model, mapping = pickle.load(file)

        if model is not None:
            return ASLClassificationModel(model, mapping)

        raise Exception("Model not loaded correctly!")

    def __init__(self, model, mapping):
        self.model = model
        self.mapping = mapping

    def predict(self, feature):
        return self.mapping[self.model.predict(feature.reshape(1, -1)).item()]