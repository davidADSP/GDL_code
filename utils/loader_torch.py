"""
TO DO:
1. make the MNIST save path static to a dir (e.g. in cache)
"""
import torchvision
import torchvision.datasets as datasets

def load_mnist():
    """
    get MNIST data loader, both train set and test set
    store MNIST dataset in ./data/MNIST folder
    """
    transform = torchvision.transforms.Compose([
        # by default div 255 
        torchvision.transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root = './data', train = True, 
                                  transform = transform, download = True)
    test_dataset = datasets.MNIST(root = './data', train = False, 
                                 transform = transform, download = True)
    return train_dataset, test_dataset


