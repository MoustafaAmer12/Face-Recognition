from sklearn.preprocessing import LabelEncoder
class encoder:
    def __init__(self, y_train):
        self.y_train = y_train
    def encode(self):
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(self.y_train)
        print(f"real: {self.y_train}\nencoded: {y_train_encoded}")
