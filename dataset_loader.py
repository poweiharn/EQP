from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split

from dataset import PascalVOC_Dataset
import torch

PascalVOC_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']


def encode_labels(target):
    ls = target['annotation']['object']
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            return torch.LongTensor([PascalVOC_categories.index(ls['name'])]).flatten()
    else:
        for i in range(len(ls)):
            return torch.LongTensor([PascalVOC_categories.index(ls[i]['name'])]).flatten()


def get_data_set(configs):
    train_loader = None
    val_loader = None
    test_loader = None
    if configs.dataset == "cifar100":
        cifar100_total_train_length = 50000
        # cifar100_total_train = 50000
        # cifar100_total_train = 128
        cifar100_total_test_length = 10000
        cifar100_total_test = 10000

        cifar100_val_length = 10000

        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        # data transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(configs.img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = datasets.CIFAR100(root='./data/cifar100',
                                     train=True,
                                     download=True,
                                     transform=transform_train)

        configs.num_class = len(trainset.classes)

        trainset, valset = random_split(trainset, [cifar100_total_train_length - cifar100_val_length, (
            cifar100_val_length)])
        # get dataloader
        train_sampler = RandomSampler(trainset)

        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=configs.batch_size,
                                  num_workers=configs.num_workers,
                                  pin_memory=configs.pin_memory)
        print("CIFAR100 Dataset Train Size: ", len(train_loader.dataset))

        val_loader = DataLoader(valset,
                                batch_size=configs.eval_batch_size,
                                num_workers=configs.num_workers,
                                pin_memory=configs.pin_memory)
        print("CIFAR100 Dataset Valid Size: ", len(val_loader.dataset))

        # data transforms
        transform_test = transforms.Compose([
            transforms.Resize((configs.img_size, configs.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        testset = datasets.CIFAR100(root='./data/cifar100',
                                    train=False,
                                    download=True,
                                    transform=transform_test)
        testset, val_dataset = random_split(testset, [cifar100_total_test, (
                cifar100_total_test_length - cifar100_total_test)])
        # get dataloader
        test_sampler = SequentialSampler(testset)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=configs.eval_batch_size,
                                 num_workers=configs.num_workers,
                                 pin_memory=configs.pin_memory)

        print("CIFAR100 Dataset Test Size: ", len(test_loader.dataset))

    elif configs.dataset == "cifar10":
        # num_workers = 8,
        # train_batch_size = 128,
        # eval_batch_size = 256
        cifar10_total_train_length = 50000
        cifar10_val_length = 10000

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        train_set = datasets.CIFAR10(root="./data/cifar10",
                                     train=True,
                                     download=True,
                                     transform=train_transform)
        configs.num_class = len(train_set.classes)

        train_set, valset = random_split(train_set, [cifar10_total_train_length - cifar10_val_length, (
            cifar10_val_length)])

        test_set = datasets.CIFAR10(root="./data/cifar10",
                                    train=False,
                                    download=True,
                                    transform=test_transform)

        train_sampler = torch.utils.data.RandomSampler(train_set)
        test_sampler = torch.utils.data.SequentialSampler(test_set)

        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=configs.batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=configs.num_workers)
        print("CIFAR10 Dataset Train Size: ", len(train_loader.dataset))

        val_loader = DataLoader(valset,
                                batch_size=configs.eval_batch_size,
                                num_workers=configs.num_workers,
                                pin_memory=configs.pin_memory)
        print("CIFAR10 Dataset Valid Size: ", len(val_loader.dataset))

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=configs.eval_batch_size,
                                                  sampler=test_sampler,
                                                  num_workers=configs.num_workers)
        print("CIFAR10 Dataset Test Size: ", len(test_loader.dataset))



    elif configs.dataset == "svhn":
        svhn_total_train_length = 73257
        svhn_val_length = 18315
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4380, 0.4440, 0.4730),
                                 std=(0.1751, 0.1771, 0.1744))])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        train_set = datasets.SVHN(root="./data/svhn", split='train', download=True,
                                  transform=train_transform)

        configs.num_class = 10
        train_set, valset = random_split(train_set, [svhn_total_train_length - svhn_val_length, (
            svhn_val_length)])

        test_set = datasets.SVHN(root="./data/svhn", split='test', download=True,
                                 transform=test_transform)

        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=configs.batch_size,
                                                   shuffle=True,
                                                   num_workers=configs.num_workers)
        print("SVHN Dataset Train Size: ", len(train_loader.dataset))

        val_loader = DataLoader(valset,
                                batch_size=configs.eval_batch_size,
                                shuffle=False,
                                num_workers=configs.num_workers)
        print("SVHN Dataset Valid Size: ", len(val_loader.dataset))

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=configs.eval_batch_size,
                                                  shuffle=False,
                                                  num_workers=configs.num_workers)
        print("SVHN Dataset Test Size: ", len(test_loader.dataset))





    elif configs.dataset == "mnist":
        mnist_total_train_length = 60000
        mnist_val_length = 15000
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.Grayscale(3),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.5), std=(0.5))])
        test_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.Grayscale(3),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.50), std=(0.5))])

        train_dataset = datasets.MNIST(root='./data/mnist', train=True,
                                       download=True,
                                       transform=train_transform)

        configs.num_class = len(train_dataset.classes)
        train_set, val_set = random_split(train_dataset, [mnist_total_train_length - mnist_val_length, (
            mnist_val_length)])

        test_dataset = datasets.MNIST(root='./data/mnist', train=False,
                                      download=True,
                                      transform=test_transform)

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=configs.batch_size,
                                                   shuffle=True,
                                                   num_workers=configs.num_workers)
        print("MNIST Dataset Train Size: ", len(train_loader.dataset))

        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=configs.eval_batch_size,
                                                 shuffle=False,
                                                 num_workers=configs.num_workers)
        print("MNIST Dataset Valid Size: ", len(val_loader.dataset))

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=configs.eval_batch_size,
                                                  shuffle=False,
                                                  num_workers=configs.num_workers)
        print("MNIST Dataset Test Size: ", len(test_loader.dataset))


    elif configs.dataset == "PascalVOC":
        # Define the transformation to apply to the data
        # Imagnet values
        mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
        std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

        #    mean=[0.485, 0.456, 0.406]
        #    std=[0.229, 0.224, 0.225]
        data_dir = "./data/PascalVOC"
        download_data = False
        configs.num_class = 20

        transformations = transforms.Compose([transforms.Resize((32, 32)),
                                              #                                      transforms.RandomChoice([
                                              #                                              transforms.CenterCrop(300),
                                              #                                              transforms.RandomResizedCrop(300, scale=(0.80, 1.0)),
                                              #                                              ]),
                                              transforms.RandomChoice([
                                                  transforms.ColorJitter(brightness=(0.80, 1.20)),
                                                  transforms.RandomGrayscale(p=0.25)
                                              ]),
                                              transforms.RandomHorizontalFlip(p=0.25),
                                              transforms.RandomRotation(25),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std),
                                              ])

        transformations_valid = transforms.Compose([transforms.Resize((32, 32)),
                                                    transforms.CenterCrop(32),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=mean, std=std),
                                                    ])

        # Create train dataloader
        dataset_train = PascalVOC_Dataset(data_dir,
                                          year='2007',
                                          image_set='train',
                                          download=download_data,
                                          transform=transformations,
                                          target_transform=encode_labels)

        train_loader = DataLoader(dataset_train, batch_size=configs.batch_size, num_workers=configs.num_workers,
                                  shuffle=True)

        # Create validation dataloader
        dataset_test = PascalVOC_Dataset(data_dir,
                                         year='2007',
                                         image_set='test',
                                         download=download_data,
                                         transform=transformations_valid,
                                         target_transform=encode_labels)

        test_loader = DataLoader(dataset_test, batch_size=configs.eval_batch_size, num_workers=configs.num_workers)

        print("PASCAL Dataset Train Size: ", len(train_loader.dataset))
        print("PASCAL Dataset Test Size: ", len(test_loader.dataset))

    return train_loader, val_loader, test_loader



