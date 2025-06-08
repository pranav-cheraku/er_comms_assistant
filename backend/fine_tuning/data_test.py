import pandas as pd

train = pd.read_csv("../data/MTS-Dialog-TrainingSet.csv")
print(train['section_text'][0])