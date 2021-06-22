import pickle


def save(model, fileName):
    filename = fileName
    pickle.dump(model, open(filename, 'wb'))


def load(fileName):
    filename = fileName
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
