import argparse

parser = argparse.ArgumentParser()


# Hardware specifications 硬件信息
parser.add_argument('--n_threads', type=int, default=2,
                    help='number of threads for data loading')
parser.add_argument('--cpu', type=bool, default=False,
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                    help='list of gpu ids')


# Data specifications
parser.add_argument('--data_dir', type=str, default='./data',
                    help='dataset directory')
parser.add_argument('--modal1', type=str, default='T1',
                    help='name of modal 1')
parser.add_argument('--modal2', type=str, default='DTI',
                    help='name of modal 2')
parser.add_argument('--train_ratio',type=float, default=0.8,
                    help='train_ratio')
parser.add_argument('--val_ratio',type=float, default=0.05,
                    help='val_ratio')
parser.add_argument('--test_ratio',type=float, default=0.15,
                    help='test_ratio')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Model specifications
parser.add_argument('--model', default='demo',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default= 'model/xxxx.pt',
                    help='pre-trained model directory')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=40,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from the snapshot, and the start_epoch')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*MSE',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--save', type=str, default='firefly',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_every', type=int, default=200,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_every', type=int, default=30,
                    help='how many batches to wait before logging training status')

args = parser.parse_args()


for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False