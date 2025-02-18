import copy
import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from inc_net import ResNetCosineIncrementalNet,SimpleVitNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy
import torch.nn.functional as F
from utils.maxsplit import maxsplit
import json
from tqdm import *
import pandas as pd
from utils.predictitid import *

num_workers = 8
def select_random_pairs(tensor1, tensor2, ratio):
    num_elements = tensor1.size(0)
    num_selected = int(ratio * num_elements)
    random_indices = torch.randperm(num_elements)[:num_selected]
    selected_tensor1 = tensor1[random_indices]
    selected_tensor2 = tensor2[random_indices]
    return selected_tensor1, selected_tensor2

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self.ptask = 0
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments=[]
        self._network = None

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        acc_total,grouped = self._evaluate(y_pred, y_true)
        return acc_total,grouped,y_pred[:,0],y_true

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1] 
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  
    
    def _evaluate(self, y_pred, y_true):
        ret = {}
        acc_total,grouped = accuracy(y_pred.T[0], y_true, self._known_classes,self.class_increments)
        return acc_total,grouped 
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"]!='ncm':
            if args["model_name"]=='adapter' and '_adapter' not in args["convnet_type"]:
                raise NotImplementedError('Adapter requires Adapter backbone')
            if args["model_name"]=='ssf' and '_ssf' not in args["convnet_type"]:
                raise NotImplementedError('SSF requires SSF backbone')
            if args["model_name"]=='vpt' and '_vpt' not in args["convnet_type"]:
                raise NotImplementedError('VPT requires VPT backbone')
            if 'resnet' in args['convnet_type']:
                self._network = ResNetCosineIncrementalNet(args, True)
                self._batch_size=128
            else:
                self._network = SimpleVitNet(args, True)
                self._batch_size= args["batch_size"]
            self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
            self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        else:
            self._network = SimpleVitNet(args, True)
            self._batch_size= args["batch_size"]
        self.args=args
        self.metric = np.zeros((10,10))

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def update_ptask(self):

        self.ptask += 1
    
    def replace_fc(self,trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            self._network.fc.use_RP=True
            if self.args['M']>0:
                self._network.fc.W_rand=self.W_rand
            else:
                self._network.fc.W_rand=None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y = target2onehot(label_list,self.total_classnum)

        if self.args['use_RP']:
            if self.args['M'] > 0:  Features_h = torch.nn.functional.relu(Features_f @ self._network.fc.W_rand.cpu())
            else:   Features_h=Features_f
            for class_index in np.unique(label_list):
                data_index = (label_list==class_index).nonzero().squeeze(-1)
                Features_class = Features_f[data_index]
                class_prototype = Features_class.mean(0)
                self.cp[class_index] = class_prototype
                cos_similarities = []
                for feature in Features_class:
                    cos_similarities.append(F.cosine_similarity(feature, self.cp[class_index], dim=0))
                self.mean_dis[class_index] = torch.mean(torch.tensor(cos_similarities))
            this_data = []
            for i in trange(len(Features_f), desc="Creating train data"):
                my_data = []
                for j in np.unique(label_list):
                    my_data.append(torch.norm(Features_f[i] - self.cp[j],p = 2).item())
                this_data.append(my_data)
            this_data = np.array(this_data)
            df = pd.DataFrame(this_data)
            for class_index in np.unique(label_list):
                for pt in range(self.ptask):
                    if len(self.p2c[pt]) >= self.args['k']:
                        self.update_wo[pt] = False
                        continue
                    flag = True
                    for pt_c in self.p2c[pt]:
                        if F.cosine_similarity(self.cp[class_index],self.cp[pt_c],dim = 0) > self.mean_dis[pt_c]:
                            flag = False
                            break
                    if flag: 
                        self.mp[class_index] = pt
                        self.p2c[pt].append(class_index)
                        self.mask[self._cur_task].add(pt)
                        break
            last_class = []
            for class_index in np.unique(label_list):
                if class_index in self.mp:  pass
                else:   last_class.append(class_index)
            if last_class:    
                edge = {}
                for v in last_class:
                    for u in last_class:
                        if F.cosine_similarity(self.cp[u],self.cp[v],dim = 0) > self.mean_dis[v]:
                            if v not in edge:   edge[v] = []
                            edge[v].append(u)
                
                split = maxsplit(last_class,edge)
                for t in range(len(split)):
                    for class_index in split[t]:
                        self.mp[class_index] = self.ptask
                        self.p2c[self.ptask].append(class_index)
                        self.mask[self._cur_task].add(self.ptask)
                    self.add_Ptask()
            train_label = [self.mp[i.item()] for i in label_list]
            df['label'] = train_label
            df['orginal_label'] = [i.item() for i in label_list]
            self.traing_data.append(df)
            Features_each_ptask = {}
            label_each_ptask = {}
            print(f"Class Groups: {self.p2c}")
            for pt in range(self.ptask):
                data_index = []
                for index, label in enumerate(label_list):
                    if self.mp[label.item()] == pt:
                        data_index.append(index)
                data_index = torch.tensor(data_index).long()
                Features_each_ptask[pt] = Features_h[data_index]
                label_each_ptask[pt] = Y[data_index]
                self.Q[pt] = self.Q[pt] + Features_each_ptask[pt].T @ label_each_ptask[pt]
                self.G[pt] = self.G[pt] + Features_each_ptask[pt].T @ Features_each_ptask[pt]
                ridge = self.optimise_ridge_parameter(torch.cat((torch.tensor(self.reply_data[pt]), Features_each_ptask[pt]), dim=0),torch.cat((torch.tensor(self.reply_label[pt]) , label_each_ptask[pt]),dim = 0))
                s1,s2 = select_random_pairs(Features_each_ptask[pt],label_each_ptask[pt],0.2)
                self.reply_data[pt].extend(s1.tolist())
                self.reply_label[pt].extend(s2.tolist())
                self.Wo[pt] = torch.linalg.solve(self.G[pt]+ridge*torch.eye(self.G[pt].size(dim=0)),self.Q[pt]).T
            
        
    
    def test_for_all_task(self,now_task,data_manager):
        ave_acc = 0
        all_train_data = pd.concat(self.traing_data, ignore_index=True)
        models, best_a, best_b, best_c = train_model(all_train_data)
        total_Wo = torch.zeros_like(self.Wo[0]) 
        for i in range(len(self.Wo)-1):
            total_Wo = total_Wo + self.Wo[i]
        for past_task in range(now_task+1):
            test_dataset = data_manager.get_dataset(np.arange(past_task*self.args["increment"],(past_task+1)*self.args["increment"]), source="test", mode="test" )
            testloader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
            Features_f = []
            label_list = []    
            with torch.no_grad():
                for i, batch in enumerate(tqdm(testloader, desc="Processing test data")):
                    (_,data,label)=batch
                    data=data.cuda()
                    label=label.cuda()
                    embedding = self._network.convnet(data)
                    Features_f.append(embedding.cpu())
                    label_list.append(label.cpu())
            Features_f = torch.cat(Features_f, dim=0)
            label_list = torch.cat(label_list, dim=0)
            this_data = []
            for i in trange(len(Features_f), desc="Creating test data"):
                my_data = []
                for j in np.unique(label_list):
                    my_data.append(torch.norm(Features_f[i] - self.cp[j],p = 2).item())
                this_data.append(my_data)
            this_data = np.array(this_data)
            df = pd.DataFrame(this_data)
            test_label = [self.mp[i.item()] for i in label_list]
            df['label'] = test_label
            df['orginal_label'] = [i.item() for i in label_list]
            task_id, acc = test_model(models,df,best_a,best_b,best_c)
            y_pred = []
            if self.args['M'] > 0:  Features_h = torch.nn.functional.relu(Features_f @ self._network.fc.W_rand.cpu())
            else:   Features_h=Features_f
            for i in range(Features_h.shape[0]):
                score = Features_h[i]@total_Wo.T + 1.5*Features_h[i]@self.Wo[int(task_id[i])].T
                y_pred.append(score.argmax(dim = 0))
            accuracy = sum([y_pred[i] == label_list[i] for i in range(len(y_pred))])/len(y_pred)
            ave_acc += accuracy
            self.metric[now_task][past_task] = accuracy
            print(f"Task {now_task}, Backtest {past_task}, Accuracy {accuracy}, ID Accuracy {acc}")
        print(f"Now Task {now_task}, Average Accuracy {ave_acc/(now_task+1)}")

    def after_all_task(self):
        np.save('./mat/rand.npy',self._network.fc.W_rand.cpu())
        for pt in range(self.ptask):
            np.save(f'./mat/Wo_{pt}.npy',self.Wo[pt])


    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
    
    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)
        if self.args['use_RP']:
            del self._network.fc
            self._network.fc=None
        self._network.update_fc(self._classes_seen_so_far)
        if self.is_dil == False:
            logging.info("Starting CIL Task {}".format(self._cur_task+1))
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._classes_seen_so_far-1))
        self.class_increments.append([self._known_classes, self._classes_seen_so_far-1])
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
        train_dataset_for_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="test", )
        self.train_loader_for_CPs = DataLoader(train_dataset_for_CPs, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._classes_seen_so_far), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_CPs)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def freeze_backbone(self,is_first_session=False):
        if 'vit' in self.args['convnet_type']:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False
        else:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False

    def show_num_params(self,verbose=False):
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

    def _train(self, train_loader, test_loader, train_loader_for_CPs):
        self._network.to(self._device)
        if self._cur_task == 0 and self.args["model_name"] in ['ncm','joint_linear']:
             self.freeze_backbone()
        if self.args["model_name"] in ['joint_linear','joint_full']: 
            if self.args["model_name"] =='joint_linear':
                assert self.args['body_lr']==0.0
            self.show_num_params()
            optimizer = optim.SGD([{'params':self._network.convnet.parameters()},{'params':self._network.fc.parameters(),'lr':self.args['head_lr']}], momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000])
            logging.info("Starting joint training on all data using "+self.args["model_name"]+" method")
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.show_num_params()
        else:
            if self._cur_task == 0 and self.dil_init==False:
                if 'ssf' in self.args['convnet_type']:
                    self.freeze_backbone(is_first_session=True)
                if self.args["model_name"] != 'ncm':
                    self.show_num_params()
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
                    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                    logging.info("Starting PETL training on first task using "+self.args["model_name"]+" method")
                    self._init_train(train_loader, test_loader, optimizer, scheduler)
                    self.freeze_backbone()
                if self.args['use_RP'] and self.dil_init==False:
                    self.setup_RP()
            if self.is_dil and self.dil_init==False:
                self.dil_init=True
                self._network.fc.weight.data.fill_(0.0)
            self.replace_fc(train_loader_for_CPs)
            self.show_num_params()
        
    
    def setup_RP(self):
        self.initiated_G=False
        self._network.fc.use_RP=True
        self.mp = {}
        self.cp = {}
        self.mean_dis = {}
        self.p2c = {}
        self.p2c[self.ptask] = []
        if self.args['M']>0:
            M = self.args['M']
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(device='cuda')) #num classes in task x M
            self._network.fc.reset_parameters()
            self._network.fc.W_rand = torch.randn(self._network.fc.in_features,M).to(device='cuda')
            self.W_rand = copy.deepcopy(self._network.fc.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            M = self._network.fc.in_features #this M is L in the paper
        self.Q = [torch.zeros(M,self.total_classnum)]
        self.G = [torch.zeros(M,M)]
        self.Wo = [torch.zeros(M,self.total_classnum)]
        self.mask = [set() for _ in range(100)]
        self.update_wo = [True for _ in range(100)]
        self.traing_data = []
        self.reply_data = [[] for _ in range(100)]
        self.reply_label = [[] for _ in range(100)]

    def add_Ptask(self):
        if self.args['M']>0:
            M = self.args['M']
        else:
            M = self._network.fc.in_features
        self.Q.append(torch.zeros(M,self.total_classnum))
        self.G.append(torch.zeros(M,M))
        self.Wo.append(torch.zeros(M,self.total_classnum))
        self.ptask += 1
        self.p2c[self.ptask] = []
        self.mask.append(set())

