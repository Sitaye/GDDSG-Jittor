import copy
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch

from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from GDDSG import Learner
def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    ave_accs=[]
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        ave_acc=_train(args)
        ave_accs.append(ave_acc)
    return ave_accs


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        ' ',
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info('Starting new run')
    _set_random()
    _set_device(args)
    print_args(args)

    model = Learner(args)
    model.show_num_params()
    model.dil_init=False
    if args['dataset']=='core50':
        ds='core50_s1'
        dil_tasks=['s1','s2','s4','s5','s6','s8','s9','s11']
        num_tasks=len(dil_tasks)
        model.is_dil=True
    elif args['dataset']=='cddb':
        ds='cddb_gaugan'
        dil_tasks=['gaugan','biggan','wild','whichfaceisreal','san']
        num_tasks=len(dil_tasks)
        model.topk=2
        model.is_dil=True
    elif args['dataset']=='domainnet':
        ds='domainnet_real'
        dil_tasks=['real','quickdraw','painting','sketch','infograph','clipart']
        num_tasks=len(dil_tasks)
        model.is_dil=True
    else:
        model.is_dil=False
        data_manager = DataManager(
            args['dataset'],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            use_input_norm=args["use_input_norm"]
        )
        num_tasks=data_manager.nb_tasks
    logging.info("Pre-trained network parameters: {}".format(count_parameters(model._network)))
    num_tasks = 10
    for task in range(num_tasks):
        if model.is_dil:
            data_manager = DataManager(
                args["dataset"]+'_'+dil_tasks[task],
                args["shuffle"],
                args["seed"],
                args["init_cls"],
                args["increment"],
                use_input_norm=args["use_input_norm"]
            )
            model._cur_task=-1
            model._known_classes = 0
            model._classes_seen_so_far = 0
        model.incremental_train(data_manager)
        model.after_task()
        model.test_for_all_task(task,data_manager)
    np.save(f"result.npy", model.metric)
    logging.info('Finishing run')
    logging.info('')
    return 0

def save_results(args,top1_total,ave_acc,model,classes_df):
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    output_df=pd.DataFrame()
    output_df['top1_total']=top1_total
    output_df['ave_acc']=ave_acc
    output_df.to_csv('./results/'+args['dataset']+'_publish_'+str(args['ID'])+'.csv')

    if not os.path.exists('./results/class_preds/'):
        os.makedirs('./results/class_preds/')
    classes_df.to_csv('./results/class_preds/'+args['dataset']+'_class_preds_publish_'+str(args['ID'])+'.csv')

def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus

def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
