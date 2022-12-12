# Authors: Marcin Żmuda-Trzebiatowski and Jakub Cirocki
# Example: https://github.com/s20501/NAI/blob/main/zjazd_5/examples
#
# Neural training based on TensorFlow framework


from trainSVM import trainSVM
from trainNetwork import trainNetwork
from trainNetworkClothes import trainNetworkClothes
from trainCifar10 import trainCifar10

if __name__ == '__main__':
    trainSVM("data/banknotes.csv")
    trainNetwork("data/banknotes.csv")
    trainCifar10()
    trainNetworkClothes()
