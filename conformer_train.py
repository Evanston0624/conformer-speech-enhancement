import torch
import torch.nn as nn
from torch.optim import NAdam
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np
class conformer_tr():
    def __init__(self, MdlPth, CPTPth, encoder_dim, num_encoder_layers, num_attention_heads):
        from conformer import Conformer
        batch_size = 1
        self.sequence_length = 300
        self.dim = 257
        self.MdlPth = MdlPth
        self.CPTPth = CPTPth
        self.Dtype = torch.float
        cuda = torch.cuda.is_available()  

        self.device = torch.device('cuda' if cuda else 'cpu')

        self.model = Conformer(num_classes=self.dim, 
                          input_dim=self.dim, 
                          encoder_dim=encoder_dim, 
                          num_encoder_layers=num_encoder_layers,
                          num_attention_heads=num_attention_heads).to(self.device)

        self.MSELoss = nn.MSELoss().to(self.device)
        # ibm
        # self.BCELoss = nn.BCELoss().to(self.device)
        # self.BCELoss = nn.BCEWithLogitsLoss().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)  # 尝试更低的学习率

        self.writer = SummaryWriter(CPTPth)
        
        # # 動態調整
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
    def train(self, Epochs, DtaLoader):
        self.num_epochs = Epochs
        self.dtaloader = DtaLoader
        Curr_epoch = 1
        Best_Loss = 1e10

        modSavInd = False
        while Curr_epoch <= self.num_epochs:
            self._train_epoch(Curr_epoch)
            self._eval_epoch(Curr_epoch)
            print('TrLoss:',self.TrLoss, 'VaLoss:',self.VaLoss)

            self.writer.add_scalar('TrLoss', self.TrLoss, Curr_epoch)
            self.writer.add_scalar('VaLoss', self.VaLoss, Curr_epoch)

            # checkpointing
            if  self.VaLoss <= Best_Loss:
                self._save_Mdl(Curr_epoch)
                Best_Loss = self.VaLoss
                modSavInd = True

            # # 動態
            self.scheduler.step(self.VaLoss)
            # 靜態
            # self.normal_scheduler.step()
            Curr_epoch += 1

        self.writer.flush()
        self.writer.close()


    def _train_epoch(self, Curr_epoch):
        self.model.train()
        self.TrLoss = 0
        pbar = tqdm(self.dtaloader['tr'], bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Epoch {str(Curr_epoch)}/{str(self.num_epochs)}')

        for itrNum, FeaDict in enumerate(pbar):
            # Forward propagate
            outputs_ibm, outputs_cln, output_lengths = self.model(FeaDict['DegFeat']['inpfeat'].type(self.Dtype).to(self.device), torch.LongTensor([self.sequence_length]))
            # print('outputs_ibm.shape:',outputs_ibm.shape)
            # print('outputs_cln.shape:',outputs_cln.shape)

            tarFea_ibm = FeaDict['TarSpec']['ibmspec'].type(self.Dtype).to(self.device)
            tarFea_cln = FeaDict['TarSpec']['rcnspec'].type(self.Dtype).to(self.device)

            # loss_1 = self.BCELoss(outputs_ibm.type(self.Dtype), tarFea_ibm.type(self.Dtype))
            loss_1 = self.MSELoss(outputs_ibm.type(self.Dtype), tarFea_ibm.type(self.Dtype))
            loss_2 = self.MSELoss(outputs_cln.type(self.Dtype), tarFea_cln.type(self.Dtype))
            loss = loss_1 + loss_2

            self.optimizer.zero_grad()
            loss.backward()

            # # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.TrLoss += loss.item()

            # pbar.set_postfix_str('loss={:^7.3f}'.format(self.TrLoss/(itrNum + 1)))
            # pbar.set_postfix_str('loss1={:^7.3f}'.format(loss_1/(itrNum + 1)), 'loss2={:^7.3f}'.format(loss_2/(itrNum + 1)))
            pbar.set_postfix({'loss1': loss_1.item(), 'loss2': loss_2.item()})

        pbar.close()
        self.TrLoss /= len(self.dtaloader['tr'])

    def _eval_epoch(self, Curr_epoch):
        self.model.eval()
        self.VaLoss = 0

        pbar = tqdm(self.dtaloader['va'], bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Val')

        for itrNum, FeaDict in enumerate(pbar):

            with torch.no_grad():
                outputs_ibm, outputs_cln, output_lengths = self.model(FeaDict['DegFeat']['inpfeat'].type(self.Dtype).to(self.device), torch.LongTensor([self.sequence_length]))
                tarFea_ibm = FeaDict['TarSpec']['ibmspec'].type(self.Dtype).to(self.device)
                tarFea_cln = FeaDict['TarSpec']['rcnspec'].type(self.Dtype).to(self.device)

                # loss_1 = self.BCELoss(outputs_ibm.type(self.Dtype), tarFea_ibm.type(self.Dtype))
                loss_1 = self.MSELoss(outputs_ibm.type(self.Dtype), tarFea_ibm.type(self.Dtype))
                loss_2 = self.MSELoss(outputs_cln.type(self.Dtype), tarFea_cln.type(self.Dtype))

            self.VaLoss += loss_1.item() + loss_2.item()

            pbar.set_postfix_str('loss={:^7.3f};'.format(self.VaLoss/(itrNum + 1)))

        pbar.close()
        self.VaLoss /= len(self.dtaloader['va'])

    def eval(self, data):
        self.model.eval()
        # data = data.to(self.device)

        for step in range(int(data.shape[1]/self.sequence_length)+1):
            inp_data = data[:, step*300:(step+1)*300, :]

             # if input len <= 300
            if inp_data.shape[1] <= 300:
                # 创建一个形状为 (1, 300, self.dim) 的全零张量
                padded_data = torch.zeros((1, 300, self.dim))

                padded_data[:, :inp_data.shape[1], :] = inp_data
                outputs_ibm, outputs_cln, output_lengths = self.model(padded_data.to(self.device), torch.LongTensor([self.sequence_length]))
                outputs_ibm = outputs_ibm[:, :inp_data.shape[1], :]
                outputs_cln = outputs_cln[:, :inp_data.shape[1], :]
            else:
                # 如果输入数据的长度大于等于 300，则不进行填充
                padded_data = inp_data
                outputs_ibm, outputs_cln, output_lengths = self.model(padded_data.to(self.device), torch.LongTensor([self.sequence_length]))
    
            if step == 0 :
                opt_ibm = outputs_ibm
                opt_cln = outputs_cln
            else :
                opt_ibm = torch.cat((opt_ibm, outputs_ibm), 1)
                opt_cln = torch.cat((opt_cln, outputs_cln), 1)

        return opt_ibm, opt_cln

    def _save_Mdl(self, Curr_epoch):
        print(f'Saving model to {self.MdlPth}')
        state_dict = {
            'epoch': Curr_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.VaLoss,
            }

        torch.save(state_dict, self.MdlPth+'/'+str(Curr_epoch)+'mdl.pkl')

    def _load_Mdl(self):
        checkpoint = torch.load(self.MdlPth)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.type(self.Dtype).to(self.device)

    def load_loss(self, model_path):
        checkpoint = torch.load(model_path)
        print(checkpoint['loss'])