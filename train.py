import argparse
from configs import configs
from dataset_loader import get_data_set
from train_engine import train_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Pooling Survey Args')
    parser.add_argument('--gpu', type=str, help="gpu choose, eg. '0,1,2,...' ")
    parser.add_argument('--name', type=str, default="123", help='the name of this training')
    parser.add_argument('--runs', type=int, default=1, help='the name of this training')
    parser.add_argument('--img_size', type=int, default=32, help='Resolution size')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help="learning rate decay rate")
    parser.add_argument('--warmup_epoch', type=int, default=1, help='warmup epochs')
    parser.add_argument('--epoch', type=int, default=3, help='total epochs')
    parser.add_argument('--seed', type=int, default=2, help='seed')
    parser.add_argument('--save_epoch', type=int, default=20, help="save model after every 20 epoch")
    parser.add_argument('--eval_every_epoch', action='store_true', default=True, help='evaluate the model every epoch')
    parser.add_argument('--model', type=str,
                        choices=['vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn'], default='vgg16_bn',help='choose model')
    parser.add_argument('--dataset', type=str,
                        choices=[
                            'cifar100',
                            'cifar10',
                            'mnist',
                            'svhn'
                        ], default='cifar100',help='choose dataset')
    parser.add_argument('--pooling', type=str,
                        choices=[
                            'M',
                            'A',
                            'X',
                            'S',
                            'T'
                        ], default='M',help='choose one pooling method to use')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args_dict = configs.parse_to_dict(args)
    configs.add_args(args_dict)
    configs.epoch = args.epoch + args.warmup_epoch
    configs.training_init()
    configs.path_init()

    train_loader, valid_loader, test_loader = get_data_set(configs)

    result = {}
    for run in range(args.runs):
        sim_data = train_engine(configs, train_loader, valid_loader, test_loader)
        result[str(run + 1)] = sim_data
    f = open(configs.result_log_dir+"/result.txt", "w")
    f.write(str(result))
    f.close()
