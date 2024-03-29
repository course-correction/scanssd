from args import parse_train_args
from data_loaders import *
from data_loaders.gtdb_iterable import GTDBDetection, worker_init_fn
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
from utils import helpers
import logging
import time
import datetime
# from torchviz import make_dot

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def train(args):

    cfg = exp_cfg[args.cfg]
    dataset = GTDBDetection(args, args.training_data, split='train',
                            transform=SSDAugmentation(cfg['min_dim']), mean=MEANS)

    # TBWriter
    writer = SummaryWriter('src/TensorBoard_Logs/' + args.exp_name)

    # if args.visdom:
    #     import visdom
    #     viz = visdom.Visdom()

    gpus = [str(gpu_id) for gpu_id in args.gpu]
    gpus = ','.join(gpus)
    devices = torch.device('cuda:'+gpus)

    # if args.cuda:
    #     gpu_id = helpers.get_freer_gpu()
    #     logging.debug('Using GPU with id ' + str(gpu_id))
    #     torch.cuda.set_device(gpu_id)

    torch.manual_seed(65)
    ssd_net = build_ssd(args, 'train', cfg, devices, cfg['min_dim'], cfg['num_classes'])
    ssd_net = nn.DataParallel(ssd_net, device_ids=args.gpu)
    logging.debug(ssd_net)

    ct = 0
    # freeze first few layers

    for child in ssd_net.modules():#.vgg.children():
        if ct >= args.layers_to_freeze:
            break
        #
        child.requires_grad = False
        ct += 1
        # print(child)


    if args.resume:
        logging.debug('Resuming training, loading {}...'.format(args.resume))
        ssd_net.module.load_state_dict(torch.load(args.resume))
    else:
        vgg_weights = torch.load(os.path.join("src", "base_weights", args.basenet))
        logging.debug('Loading base network...')
        ssd_net.module.vgg.load_state_dict(vgg_weights)

    #visualize(ssd_net, gpu_id)

    if not args.resume:
        logging.debug('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.module.extras.apply(weights_init)
        ssd_net.module.loc.apply(weights_init)
        ssd_net.module.conf.apply(weights_init)

        # Saving random initialized weights
        torch.save(ssd_net.module.state_dict(),
                   os.path.join(
                       'src','weights_' + args.exp_name, 'initial_' + str(args.model_type) + args.dataset + '.pth'))

    optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    # optimizer = optim.Adam(ssd_net.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=83000,
                                                           eta_min=1e-7, last_epoch=-1)

    #args, cfg, overlap_thresh, bkg_label, neg_pos
    #criterion = MultiBoxLoss(args, cfg, 0.5, 0, 3)
    criterion = MultiBoxLoss(args, cfg, args.pos_thresh, 0, 3)

    if args.cuda:

        ssd_net = ssd_net.to(devices)
        cudnn.benchmark = True

    ssd_net.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    min_total_loss = float('inf')
    logging.debug('Loading the dataset...')

    epoch_size = dataset.tot // args.batch_size
    logging.debug(f'Num Batches per epoch {epoch_size}')

    logging.debug('Training SSD on:' + args.dataset)
    logging.debug('Using the specified args:')
    logging.debug(args)


    # if args.visdom:
    #     vis_title = args.exp_name
    #     vis_legend = ['Location Loss', 'Confidence Loss', 'Total Loss']
    #     iter_plot = create_vis_plot('Iteration', 'Loss', viz, 'Training ' + vis_title, vis_legend)
    #     epoch_plot = create_vis_plot('Epoch', 'Loss', viz, 'Training ' + vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=detection_collate)

    logging.debug('Training set size is {}'.format(dataset.tot))
    logging.debug('Creating tensorboard graph')

    # Create one batch iterator for Tensorboard
    batch_iterator = iter(data_loader)
    images, targets = next(batch_iterator)
    images = images.cuda()
    writer.add_graph(ssd_net, images)

    # Run training
    for epoch in range(cfg['max_epochs']):
        logging.debug(f'Epoch: {epoch+1}')
        for iteration, (images, targets) in enumerate(data_loader):

            t0 = time.time()

            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            # forward
            out = ssd_net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = args.alpha * loss_l + loss_c # TODO. For now alpha should be 1. While plotting alpha is assumed to be 1
            loss.backward()
            optimizer.step()


            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            t1 = time.time()

            # Log progress
            if iteration % 10 == 0:
                logging.debug('timer: %.4f sec.' % (t1 - t0))
                logging.debug('iter ' + repr(iteration) + ' || Loss: %.3f ||' % (loss.item()) +
                              ' Epoch: ' + repr(epoch+1))
                writer.add_scalar('Training_Box_Loss', loss_l, iteration+(epoch_size*epoch))
                writer.add_scalar('Training_Classification_Loss', loss_c, iteration+(epoch_size*epoch))
                writer.add_scalar('Training_Total_Loss', loss, iteration+(epoch_size*epoch))

                if args.validation_data != '':
                    # Validate data_loaders
                    validation_loss = validate(args, ssd_net, criterion, cfg)

                    if epoch == 1:
                        validation_plot = create_validation_plot(epoch, validation_loss,
                                                                 'Epoch', 'Loss', viz, 'Validating ' + vis_title,
                                                                 ['Validation'])
                    else:
                        update_validation_plot(epoch, validation_loss, viz,
                                               validation_plot, 'append')

                    if validation_loss < min_total_loss:
                        min_total_loss = validation_loss
                        torch.save(ssd_net.state_dict(),
                                   os.path.join(
                                       'weights_' + args.exp_name, 'best_ssd' + str(args.model_type) + args.dataset +
                                       repr(iteration) + '.pth'))

        scheduler.step(epoch)
        # Save checkpoint after every epoch
        logging.debug('Saving state, iter:' + str(epoch+1))
        torch.save(ssd_net.module.state_dict(),
                   os.path.join(
                       'src', 'weights_'+ args.exp_name, 'src', str(args.model_type) + args.dataset +
                       '_epoch' + str(epoch+1) + '.pth'))

def validate(args, net, criterion, cfg):

    validation_batch_size = 1
    try:
        # Turn off learning. Go to testing phase
        net.eval()

        dataset = GTDBDetection(args, args.validation_data, split='validate',
                                transform=SSDAugmentation(cfg['min_dim'], mean=MEANS))

        data_loader = data.DataLoader(dataset, validation_batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=False, collate_fn=detection_collate,
                                      pin_memory=True)

        total = len(dataset)
        done = 0
        loc_loss = 0
        conf_loss = 0

        start = time.time()

        for batch_idx, (images, targets, ids) in enumerate(data_loader):

            done = done + len(images)
            logging.debug('processing {}/{}'.format(done, total))

            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            y = net(images)  # forward pass

            loss_l, loss_c = criterion(y, targets)
            loc_loss += loss_l.item()  # data_loaders[0]
            conf_loss += loss_c.item()  # data_loaders[0]

        end = time.time()
        logging.debug('Time taken for validation ' + str(datetime.timedelta(seconds=end - start)))

        return (loc_loss + conf_loss) / (total/validation_batch_size)
    except Exception as e:
        logging.error("Could not validate", exc_info=True)
        return 0

def adjust_learning_rate(args, optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def visualize(ssd_net, gpu_id):
    x = np.zeros((300,300,3))
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.to(gpu_id,  dtype=torch.float)
    ssd_net.eval()
    y = ssd_net(x)
    make_dot(y[0], params=dict(ssd_net.named_parameters())).render(filename='ssd_net')


def xavier(param):
    init.xavier_uniform_(param, gain=0.5)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def create_validation_plot(epoch, validation_loss, _xlabel, _ylabel, viz, _title, _legend):
    return viz.line(
        X=torch.ones((1, 1)).cpu() * epoch,
        Y=torch.Tensor([validation_loss]).unsqueeze(0).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def create_vis_plot(_xlabel, _ylabel, viz, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, len(_legend))).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_validation_plot(epoch, validation_loss,
                           viz, window, update_type):

    viz.line(
        X=torch.ones((1, 1)).cpu() * epoch,
        Y=torch.Tensor([validation_loss]).unsqueeze(0).cpu(),
        win=window,
        update=update_type
    )


def update_vis_plot(iteration, loc, viz, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':

    args = parse_train_args()
    start = time.time()

    try:
        filepath=os.path.join(args.log_dir, args.exp_name + "_" + str(round(time.time())) + ".log")
        print('Logging to ' + filepath)
        logging.basicConfig(filename=filepath,
                            filemode='w', format='%(process)d - %(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        train(args)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

    end = time.time()
    logging.debug('Total time taken ' + str(datetime.timedelta(seconds=end - start)))
    logging.debug("Training done!")
