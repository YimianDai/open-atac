#!/work/yimian/MXEnv/bin/python

import time
import socket
import logging
import argparse
import matplotlib
import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms
from model import ResNet20V2ATAC

matplotlib.use('Agg')
gcv.utils.check_version('0.6.0')


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')

    parser.add_argument('--model', type=str, default='xxx',
                        help='atac, se')

    parser.add_argument('--blocks', type=int, default=0,
                        help='[blocks] * 3')
    parser.add_argument('--act-type', type=str, default='xxx',
                        help='relu, prelu, swish, xUnit, SpaATAC, ChaATAC, SeqATAC')
    parser.add_argument('--r', type=int, default=-1,
                        help='2')
    parser.add_argument('--act-layers', type=int, default=4,
                        help='4')

    parser.add_argument('--useGlobal', action='store_true', default=
                        False, help='useGlobal')
    parser.add_argument('--useReLU', action='store_true', default=
                        False, help='useReLU')
    parser.add_argument('--summary', action='store_true', default=
                        False, help='print parameter number')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='cifar10 or cifar100.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--save-plot-dir', type=str, default='.',
                        help='the path to save the history plot')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    batch_size = opt.batch_size
    if opt.dataset == 'cifar10':
        classes = 10
    elif opt.dataset == 'cifar100':
        classes = 100
    else:
        raise ValueError('Unknown Dataset')

    if len(mx.test_utils.list_gpus()) == 0:
        context = [mx.cpu()]
    else:
        context = [mx.gpu(int(i)) for i in opt.gpus.split(',') if i.strip()]
        context = context if context else [mx.cpu()]
    print("context: ", context)
    num_gpus = len(context)
    batch_size *= max(1, num_gpus)
    num_workers = opt.num_workers

    lr_decay = opt.lr_decay
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

    model_name = 'ResNet20_b_' + str(opt.blocks) + '_' + opt.act_type
    print("model_name", model_name)

    if model_name.startswith('cifar_wideresnet'):
        kwargs = {'classes': classes, 'drop_rate': opt.drop_rate}
    else:
        kwargs = {'classes': classes}

    # scenario = 'ATAC'

    # main config
    layers = [opt.blocks] * 3
    channels = [x*1 for x in [16, 16, 32, 64]]
    act_type = opt.act_type  # relu, prelu, elu, selu, gelu, swish, xUnit, ChaATAC
    r = opt.r

    # spatial scope
    skernel = 3
    dilation = 1
    act_dilation = 1      # (8, 16), 4

    # ablation study
    useReLU = opt.useReLU
    useGlobal = opt.useGlobal
    asBackbone = False
    act_layers = opt.act_layers
    replace_act = 'relu'
    act_order = 'bac'  # 'pre', 'bac'

    print("model: ", opt.model)
    print("r: ", opt.r)
    if opt.model == 'atac':
        net = ResNet20V2ATAC(layers=layers, channels=channels, classes=classes,
                             act_type=act_type, r=r, skernel=skernel, dilation=dilation,
                             useReLU=useReLU, useGlobal=useGlobal, act_layers=act_layers,
                             replace_act=replace_act, act_order=act_order, asBackbone=asBackbone)

        print("layers: ", layers)
        print("channels: ", channels)
        print("act_type: ", act_type)

        print("skernel: ", skernel)
        print("dilation: ", dilation)
        print("act_dilation: ", act_dilation)

        print("useReLU: ", useReLU)
        print("useGlobal: ", useGlobal)
        print("asBackbone: ", asBackbone)
        print("act_layers: ", act_layers)
        print("replace_act: ", replace_act)
        print("act_order: ", act_order)

    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx=context)
    optimizer = 'nag'

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_period = 0

    plot_path = opt.save_plot_dir

    logging.basicConfig(level=logging.INFO)
    logging.info(opt)

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

        if opt.summary:
            net.summary(mx.nd.zeros((1, 3, 32, 32)))

        if opt.dataset == 'cifar10':
            # CIFAR10
            train_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
                batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
            val_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        elif opt.dataset == 'cifar100':
            # CIFAR100
            train_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR100(train=True).transform_first(transform_train),
                batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
            val_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR100(train=False).transform_first(transform_test),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            raise ValueError('Unknown Dataset')

        if optimizer == 'nag':
            trainer = gluon.Trainer(net.collect_params(), optimizer,
                                {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
        elif optimizer == 'adagrad':
            trainer = gluon.Trainer(net.collect_params(), optimizer,
                                    {'learning_rate': opt.lr, 'wd': opt.wd})
        elif optimizer == 'adam':
            trainer = gluon.Trainer(net.collect_params(), optimizer,
                                {'learning_rate': opt.lr, 'wd': opt.wd})
        else:
            raise ValueError('Unknown optimizer')

        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        train_history = TrainingHistory(['training-error', 'validation-error'])
        host_name = socket.gethostname()

        iteration = 0
        lr_decay_count = 0

        best_val_score = 0

        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)
            alpha = 1

            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate*lr_decay)
                lr_decay_count += 1

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])

                train_metric.update(label, output)
                name, acc = train_metric.get()
                iteration += 1

            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            name, val_acc = test(ctx, val_data)
            train_history.update([1-acc, 1-val_acc])
            train_history.plot(save_path='%s/%s_history.png'%(plot_path, model_name))

            if val_acc > best_val_score:
                best_val_score = val_acc
                # net.save_parameters('%s/%.4f-cifar-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))
                pass

            logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
                (epoch, acc, val_acc, train_loss, time.time()-tic))

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                # net.save_parameters('%s/cifar10-%s-%d.params'%(save_dir, model_name, epoch))
                pass

            if epoch == epochs-1:
                with open(opt.dataset + '_' + host_name + '_GPU_' + opt.gpus + '_best_Acc.log', 'a') as f:
                    f.write('best Acc: {:.4f}\n'.format(best_val_score))

        print("best_val_score: ", best_val_score)
        if save_period and save_dir:
            # net.save_parameters('%s/cifar10-%s-%d.params'%(save_dir, model_name, epochs-1))
            pass

    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.num_epochs, context)


if __name__ == '__main__':
    main()
