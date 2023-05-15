import os
import torch
from torch.utils.data import Subset, DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN, MNIST, USPS
from datasets.celeba import CelebA
#from datasets.ffhq import FFHQ
#from datasets.gaussian import Gaussian
from torch.utils.data import Subset, TensorDataset
import numpy as np
from torchvision.datasets import ImageFolder

def get_random_colored_images(images, seed = 0x000000):
    np.random.seed(seed)
    
    images = 0.5*(images + 1)
    size = images.shape[0]
    colored_images = []
    hues = 360*np.random.rand(size)
    
    for V, H in zip(images, hues):
        V_min = 0
        
        a = (V - V_min)*(H%60)/60
        V_inc = a
        V_dec = V - a
        
        colored_image = torch.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H/60) % 6
        
        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec
        
        colored_images.append(colored_image)
        
    colored_images = torch.stack(colored_images, dim = 0)
    colored_images = 2*colored_images - 1
    
    return colored_images
 
    
    
def get_dataset(config):
    
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    
    #======================================#
    if config.data.dataset.startswith('MNIST-colored'):
        
        dataset_name =  config.data.dataset.split("_")[0]
        is_colored = dataset_name[-7:] == "colored"
        
        classes = [int(number) for number in config.data.dataset.split("_")[1:]]
        if not classes:
            classes = [i for i in range(10)]
            
            
        train_set = torchvision.datasets.MNIST(root=os.path.join(config.data.exp, 'datasets', 'MNIST') , train=True,
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)
        
        
        test_set = torchvision.datasets.MNIST(root=os.path.join(config.data.exp, 'datasets', 'MNIST'), train=False,                                                    transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)
        
        
        
        train_test = []
        for dataset in [train_set, test_set]:
            data = []
            labels = []
            for k in range(len(classes)):
                data.append(torch.stack(
                    [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == classes[k]],
                    dim=0
                ))
                labels += [k]*data[-1].shape[0]
            data = torch.cat(data, dim=0)
            
            data = data.reshape(-1, 1, 32, 32)
            labels = torch.tensor(labels)
            
            if is_colored:
                data = get_random_colored_images(data)
            
            
            data = data*0.5 + 0.5
            train_test.append(TensorDataset(data, labels))
            
       
        train_set, test_set = train_test 
        
        train_dataset = train_set
        test_dataset = test_set
        
    if config.data.dataset == 'CIFAR10':
        if(config.data.random_flip):
            dataset = CIFAR10(os.path.join('datasets', 'cifar10'), train=True, download=True,
                              transform= transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor()]))
            test_dataset = CIFAR10(os.path.join('datasets', 'cifar10_test'), train=False, download=True,
                                   transform=transforms.Compose([
                                    transforms.Resize(config.data.image_size),
                                    transforms.ToTensor()]))

        else:
            dataset = CIFAR10(os.path.join('datasets', 'cifar10'), train=True, download=True,
                              transform= transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.ToTensor()]))
            test_dataset = CIFAR10(os.path.join('datasets', 'cifar10_test'), train=False, download=True,
                                   transform=transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.ToTensor()]))

    
    elif config.data.dataset in ['CELEBA-low','CELEBA-high']:
        
        if config.data.dataset == "CELEBA-high":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(140),
                torchvision.transforms.Resize((64, 64)),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Lambda(lambda x: 2 * x - 1)
            ])
        elif config.data.dataset == "CELEBA-low":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(140),
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.Resize((64, 64)),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Lambda(lambda x: 2 * x - 1)
            ])
            
        dataset = ImageFolder("/gpfs/data/gpfs0/n.gushchin/Scon/img_align_celeba", transform=transform)
        
        train_ratio = 0.45
        test_ratio = 0.1
        
        train_size = int(len(dataset) * train_ratio)
        test_size = int(len(dataset) * test_ratio)
        idx = np.arange(len(dataset))
        
        np.random.seed(0x000000); np.random.shuffle(idx)
        
        if  config.data.dataset== "CELEBA-low":
            train_idx = idx[:train_size]
        elif config.data.dataset == "CELEBA-high":
            train_idx = idx[train_size:-test_size]
        test_idx = idx[-test_size:]
        
        """
        #============================================#
        indices = np.random.permutation(len(train_idx))
        train_idx = train_idx[indices]
        #=============================================#
        """
        
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
            
            
         
         
        
    elif config.data.dataset in ['CELEBA', 'CELEBA-even', 'CELEBA-odd']:
        
        subset = "all"
        
        if(config.data.dataset == "CELEBA-even"):
            subset = "even"
            
        elif(config.data.dataset == "CELEBA-odd"):
            subset = "odd"

        if config.data.random_flip:
            dataset = CelebA(root=os.path.join('/home/mounted/LargeScaleOptimalTransport/datasets', 'celeba'), split='train',
                             
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), subset=subset,
                             download=True)
            
        else:
            dataset = CelebA(root=os.path.join('/home/mounted/LargeScaleOptimalTransport/datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]),
                             subset=subset,
                             download=True)

        test_dataset = CelebA(root=os.path.join('/home/mounted/LargeScaleOptimalTransport/datasets', 'celeba'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), subset=subset,
                              download=True)

        
        
    elif (config.data.dataset in ["CELEBA-32px", "CELEBA-32px-even", "CELEBA-32px-odd"]):
        subset = "all"
        if (config.data.dataset == "CELEBA-32px-even"):
            subset = "even"
        elif (config.data.dataset == "CELEBA-32px-odd"):
            subset = "odd"

        if config.data.random_flip:
            dataset = CelebA(root= os.path.join('/home/mounted/LargeScaleOptimalTransport/datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(32),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), subset=subset,
                             download=True)
        else:
            dataset = CelebA(root= os.path.join('/home/mounted/LargeScaleOptimalTransport/datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(32),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), subset=subset,
                             download=True)

        test_dataset = CelebA(root= os.path.join('/home/mounted/LargeScaleOptimalTransport/datasets', 'celeba'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(32),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), subset=subset,
                              download=True)


    elif(config.data.dataset == "CELEBA-8px"):
        if config.data.random_flip:
            dataset = CelebA(root=os.path.join('datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(8),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=True)
        else:
            dataset = CelebA(root=os.path.join('datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(8),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

        test_dataset = CelebA(root=os.path.join('datasets', 'celeba'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(8),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=True)

    elif config.data.dataset == 'LSUN':
        train_folder = '{}_train'.format(config.data.category)
        val_folder = '{}_val'.format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(root=os.path.join('datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                             ]))
        else:
            dataset = LSUN(root=os.path.join('datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

        test_dataset = LSUN(root=os.path.join('datasets', 'lsun'), classes=[val_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join('datasets', 'FFHQ'), transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]), resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join('datasets', 'FFHQ'), transform=transforms.ToTensor(),
                           resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    elif config.data.dataset == "MNIST":
        if config.data.random_flip:
            dataset = MNIST(root=os.path.join('datasets', 'MNIST'),
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.Resize(config.data.image_size),
                                               transforms.ToTensor()
                                           ]))
        else:
            dataset = MNIST(root=os.path.join('datasets', 'MNIST'),
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(config.data.image_size),
                                               transforms.ToTensor()
                                           ]))
        test_dataset = MNIST(root=os.path.join('datasets', 'MNIST'),
                                           train=False,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(config.data.image_size),
                                               transforms.ToTensor()
                                           ]))
    elif config.data.dataset == "USPS":
        if config.data.random_flip:
            dataset = USPS(root=os.path.join('datasets', 'USPS'),
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.Resize(config.data.image_size),
                               transforms.ToTensor()
                           ]))
        else:
            dataset = USPS(root=os.path.join('datasets', 'USPS'),
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.Resize(config.data.image_size),
                               transforms.ToTensor()
                           ]))
        test_dataset = USPS(root=os.path.join('datasets', 'USPS'),
                            train=False,
                            download=True,
                            transform=transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.ToTensor()
                            ]))
    elif config.data.dataset == "USPS-Pad":
        if config.data.random_flip:
            dataset = USPS(root=os.path.join('datasets', 'USPS'),
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.Resize(20), # resize and pad like MNIST
                                               transforms.Pad(4),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.Resize(config.data.image_size),
                                               transforms.ToTensor()
                                           ]))
        else:
            dataset = USPS(root=os.path.join('datasets', 'USPS'),
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(20), # resize and pad like MNIST
                                               transforms.Pad(4),
                                               transforms.Resize(config.data.image_size),
                                               transforms.ToTensor()
                                           ]))
        test_dataset = USPS(root=os.path.join('datasets', 'USPS'),
                                            train=False,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(20),  # resize and pad like MNIST
                                                transforms.Pad(4),
                                                transforms.Resize(config.data.image_size),
                                                transforms.ToTensor()
                                            ]))
        
        
    elif(config.data.dataset.upper() == "GAUSSIAN"):
        if(config.data.num_workers != 0):
            raise ValueError("If using a Gaussian dataset, num_workers must be zero. \
            Gaussian data is sampled at runtime and doing so with multiple workers may cause a CUDA error.")
        if(config.data.isotropic):
            dim = config.data.dim
            rank = config.data.rank
            cov = np.diag( np.pad(np.ones((rank,)), [(0, dim - rank)]) )
            mean = np.zeros((dim,))
        else:
            cov = np.array(config.data.cov)
            mean = np.array(config.data.mean)
        
        shape = config.data.dataset.shape if hasattr(config.data.dataset, "shape") else None

        dataset = Gaussian(device=args.device, cov=cov, mean=mean, shape=shape)
        test_dataset = Gaussian(device=args.device, cov=cov, mean=mean, shape=shape)

    elif(config.data.dataset.upper() == "GAUSSIAN-HD"):
        if(config.data.num_workers != 0):
            raise ValueError("If using a Gaussian dataset, num_workers must be zero. \
            Gaussian data is sampled at runtime and doing so with multiple workers may cause a CUDA error.")
        cov = np.load(config.data.cov_path)
        mean = np.load(config.data.mean_path)
        dataset = Gaussian(device=args.device, cov=cov, mean=mean)
        test_dataset = Gaussian(device=args.device, cov=cov, mean=mean)

    elif(config.data.dataset.upper() == "GAUSSIAN-HD-UNIT"):
        # This dataset is to be used when GAUSSIAN with the isotropic option is infeasible due to high dimensionality
        #   of the desired samples. If the dimension is too high, passing a huge covariance matrix is slow.
        if(config.data.num_workers != 0):
            raise ValueError("If using a Gaussian dataset, num_workers must be zero. \
            Gaussian data is sampled at runtime and doing so with multiple workers may cause a CUDA error.")
        shape = config.data.shape if hasattr(config.data, "shape") else None
        dataset = Gaussian(device=args.device, mean=None, cov=None, shape=shape, iid_unit=True)
        test_dataset = Gaussian(device=args.device, mean=None, cov=None, shape=shape, iid_unit=True)

    return train_dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X.float()

def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)