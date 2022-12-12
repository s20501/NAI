from trainSVM import trainSVM
from trainNetwork import trainNetwork


if __name__ == '__main__':
    trainSVM("data/banknotes.csv")
    trainNetwork("data/banknotes.csv")


