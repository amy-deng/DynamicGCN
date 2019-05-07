# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, itertools, random, argparse, time, datetime

os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve

import scipy.sparse as sp

from utils import *
from models import *
import shutil
import logging
import glob
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  

# Training settings
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='THAD6h', help="dataset string")
ap.add_argument('--embedding', type=str, default='thailand', help="word embedding string")
ap.add_argument('--tensorboard_log', type=str, default='', help="name of this run (use timestamp instead)")
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', default=1000, type=int, help='number of epochs to train')
ap.add_argument('--batch', type=int, default=1, help="batch size (due to sparse matrix operations)")
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters)')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate (1 - keep probability)')
ap.add_argument('--f_dim', type=int, default=100, help="feature dimensions of graph nodes") 
ap.add_argument('--n_hidden', default=7, type=int, help='number of hidden layers')
ap.add_argument('--n_class', type=int, default=1, help="number of class (default 1)") 
ap.add_argument('--check_point', type=int, default=1, help="check point")
ap.add_argument('--model', default='DynamicGCN', choices=['DynamicGCN','GCN'], help='')
ap.add_argument('--shuffle', action='store_false', default=True, help="Shuffle dataset 0/1")
ap.add_argument('--train', type=float, default=.825, help="training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.175, help="validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.0, help="testing ratio (0, 1) test file is seperated")
ap.add_argument('--fastmode', action='store_true', default=False, help='validate during training')
ap.add_argument('--mylog', action='store_false', default=True,  help='tensorboad log')
ap.add_argument('--patience', type=int, default=10,  help='patience for early stop')
ap.add_argument('--cuda', action='store_false', default=True, help='use cuda')


args = ap.parse_args()

print('--------------Parameters--------------')
print(args)
print('--------------------------------------')
np.random.seed(args.seed)


args.cuda = args.cuda and torch.cuda.is_available() 
logger.info('CUDA status %s', args.cuda)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

time_token = str(time.time()).split('.')[0] # tensorboard model
log_token = '%s_%s_%s_%s_%s' % (args.dataset, args.f_dim, args.model, time_token, args.tensorboard_log)

if args.mylog:
    tensorboard_log_dir = 'tensorboard/%s' % (log_token)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    shutil.rmtree(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

logger.info('dimension of feature %s', args.f_dim)

if args.model == 'DynamicGCN':
    train_dict, val_dict, test_dict, pretrained_emb = load_sparse_temporal_data(args.dataset, args.embedding, args.f_dim, args.train, args.val, args.test)
else:
    train_dict, val_dict, test_dict, pretrained_emb = load_dynamic_graph_data(args.dataset, args.embedding, args.f_dim, args.train, args.val, args.test)

if args.cuda:
    pretrained_emb = pretrained_emb.cuda()
logger.info('load dataset %s', args.dataset)

if args.model == 'DynamicGCN':
    model = DynamicGCN(pretrained_emb=pretrained_emb,
                            n_output=args.n_class,
                            n_hidden=args.n_hidden, #hidden layer
                            dropout=args.dropout)
else:
    model = GCN(pretrained_emb=pretrained_emb, 
                n_output=args.n_class, 
                dropout=args.dropout)

logger.info('model %s', args.model)
if args.cuda:
    model.cuda()
# optimizer and loss
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of parameters:',pytorch_total_params)

if args.embedding == 'egypy':
    class_weights = torch.FloatTensor([0.38, 0.62])
elif args.embedding == 'india':
    class_weights = torch.FloatTensor([0.37, 0.63])
elif args.embedding == 'pakistan':
    class_weights = torch.FloatTensor([0.15, 0.85])
elif args.embedding == 'germany':
    class_weights = torch.FloatTensor([0.12, 0.88])
elif args.embedding == 'turkey':
    class_weights = torch.FloatTensor([0.21, 0.79])
elif args.embedding == 'thailand':
    class_weights = torch.FloatTensor([0.38, 0.62])
elif args.embedding == 'russian':
    class_weights = torch.FloatTensor([0.33, 0.67])
else:
    class_weights = torch.FloatTensor([0.5, 0.5])
if args.cuda:
    class_weights = class_weights.cuda()


def evaluate(epoch, val_dict, log_desc='val_'):
    model.eval()
    total = 0.
    loss, prec, rec, f1, acc, auc = 0., 0., 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    batch_size = 1
    x_val, y_val, idx_val = val_dict['x'], val_dict['y'], val_dict['idx']
    for i in range(len(x_val)):
        adj = x_val[i]
        y = y_val[i]
        idx = idx_val[i]
        

        if args.cuda:
            y = y.cuda()
            idx = idx.cuda()
            if args.model == 'DynamicGCN':
                for i in range(len(adj)):
                    adj[i] = adj[i].cuda()
            else:
                adj = adj.cuda()
                
        output,_ = model(adj, idx)

        loss_train = F.binary_cross_entropy(output, y, weight=class_weights[int(y.item())])
        loss += batch_size * loss_train.item()
        y_true += y.data.tolist()
        bi_val = np.where(output.data.cpu().numpy() > 0.5, 1, 0)
        y_pred += torch.from_numpy(bi_val).tolist()
        y_score += output.data.tolist()
        total += batch_size
 

    # print(y_pred,y_true);exit()

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")  
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f Acc: %.4f", # 
            log_desc, loss / total, auc, prec, rec, f1, acc) 

    if args.mylog:
        if log_desc != 'train_':
            writer.add_scalars('data/loss', {log_desc: loss / total}, epoch + 1)
        writer.add_scalars('data/auc', {log_desc: auc}, epoch + 1)
        writer.add_scalars('data/prec', {log_desc: prec}, epoch + 1)
        writer.add_scalars('data/rec', {log_desc: rec}, epoch + 1)
        writer.add_scalars('data/f1', {log_desc: f1}, epoch + 1)
        writer.add_scalars('data/acc', {log_desc: acc}, epoch + 1)

    return prec, rec, f1, acc, auc


def train(epoch, train_dict, val_dict, test_dict):
    model.train()
    loss, total = 0., 0.
    batch_size = 1

    x_train, y_train, idx_train = train_dict['x'], train_dict['y'], train_dict['idx']

    if sys.version_info > (3, 0):
        combined = list(zip(x_train, y_train, idx_train))
        random.shuffle(combined)
        x_train[:], y_train[:], idx_train[:] = zip(*combined)
    else:
        z = zip(x_train, y_train, idx_train)
        random.shuffle(z)
        x_train, y_train, idx_train = zip(*z)

    for i in range(len(x_train)):
        adj = x_train[i]
        y = y_train[i]
        # feature = f_train[i]
        idx = idx_train[i]
        
        if args.cuda:
            y = y.cuda()
            idx = idx.cuda()
            if args.model == 'DynamicGCN':
                for i in range(len(adj)):
                    adj[i] = adj[i].cuda()
            else:
                adj = adj.cuda()
                
        optimizer.zero_grad()
        output,_ = model(adj, idx)
        loss_train = F.binary_cross_entropy(output, y, weight=class_weights[int(y.item())])
        loss += batch_size * loss_train.item()
        total += batch_size
        loss_train.backward()
        optimizer.step()

    logger.info("train loss epoch %d %f", epoch, loss / total)
    if args.mylog:
        writer.add_scalars('data/loss', {'train_': loss / total}, epoch + 1)

    if not args.fastmode:
        if (epoch + 0) % args.check_point == 0:
            logger.info("epoch %d, checkpoint!", epoch)
            if args.val > 0.:
                # evaluate(epoch, train_dict, log_desc='train_')
                prec, rec, f1, acc, auc = evaluate(epoch, val_dict, log_desc='val_')
            else:
                evaluate(epoch, train_dict, log_desc='train_')
                prec, rec, f1, acc, auc = evaluate(epoch, test_dict, log_desc='test_')
    return acc

# Train model
t_total = time.time()
logger.info("training...")

# if args.mylog:
# model sub folder
model_dir = 'model/%s' % (log_token)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


bad_counter = 0
best_epoch = 0
best_acc = 0. 
for epoch in range(args.epochs):
    cur_acc = train(epoch, train_dict, val_dict, test_dict)
    # if args.mylog:
    model_file = '%s/%s.pkl' % (model_dir, epoch)
    torch.save(model.state_dict(), model_file)

    if cur_acc > best_acc:
        best_acc = cur_acc
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1
    if bad_counter == args.patience:
        break

# remove other models
files = glob.glob(model_dir+'/*.pkl')
for file in files:
    filebase = os.path.basename(file)
    epoch_nb = int(filebase.split('.')[0])
    if epoch_nb != best_epoch:
        os.remove(file)

logger.info("Training Finished!")
logger.info("optimization Finished!")
logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))
logger.info("Load best model and test......")
logger.info("Best epoch {}".format(best_epoch))
model.load_state_dict(torch.load(model_dir+'/{}.pkl'.format(best_epoch)))

logger.info("testing...")
evaluate(epoch+1, test_dict, log_desc='test_')


if args.mylog:
    writer.export_scalars_to_json(tensorboard_log_dir+"/all_scalars.json")
    writer.close()

print(args)
print(log_token)