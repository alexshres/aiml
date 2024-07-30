from ucimlrepo import fetch_ucirepo

spambase = fetch_ucirepo(id=94)

# data as pandas dataframe
features = spambase.data.features
target = spambase.data.targets


