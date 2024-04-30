import pickle

with open("test.p", "rb") as file:
    loaded_data = pickle.load(file)

print(loaded_data)