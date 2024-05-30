import torch
import torch.nn as nn
from torch.optim import NAdam
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

class conformer_tr():
    def __init__(self, MdlPth, CPTPth):
        from conformer import Conformer
        batch_size = 1
        self.sequence_length = 300
        dim = 257
        self.MdlPth = MdlPth
        self.CPTPth = CPTPth
        self.Dtype = torch.float
        cuda = torch.cuda.is_available()  

        self.device = torch.device('cuda' if cuda else 'cpu')

        self.model = Conformer(num_classes=dim, 
                          input_dim=dim, 
                          encoder_dim=64, 
                          num_encoder_layers=4).to(self.device)

        self.MSELoss = nn.MSELoss().to(self.device)
        self.optimizer = NAdam(self.model.parameters(), lr=1e-3)
        self.writer = SummaryWriter(CPTPth)
        self.normal_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.95)

    def train(self, Epochs, DtaLoader):
        self.num_epochs = Epochs
        self.dtaloader = DtaLoader
        Curr_epoch = 1
        Best_Loss = 1e10

        modSavInd = False
        while Curr_epoch <= self.num_epochs:
            self._train_epoch(Curr_epoch)
            self._val_epoch(Curr_epoch)
            print('TrLoss:',self.TrLoss)
            print('VaLoss:',self.VaLoss)

            self.writer.add_scalar('TrLoss', self.TrLoss, Curr_epoch)
            self.writer.add_scalar('VaLoss', self.VaLoss, Curr_epoch)

            # checkpointing
            if  self.VaLoss <= Best_Loss:
                self._save_Mdl(Curr_epoch)
                Best_Loss = self.VaLoss
                modSavInd = True

            self.normal_scheduler.step()
            Curr_epoch += 1

        self.writer.flush()
        self.writer.close()


    def _train_epoch(self, Curr_epoch):
        self.TrLoss = 0
        pbar = tqdm(self.dtaloader['tr'], bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Epoch {str(Curr_epoch)}/{str(self.num_epochs)}')

        for itrNum, FeaDict in enumerate(pbar):
            # Forward propagate
            outputs, output_lengths = self.model(FeaDict['DegFeat']['inpfeat'].type(self.Dtype).to(self.device), torch.LongTensor([self.sequence_length]))
            tarFea = TarFea = FeaDict['TarSpec']['rcnspec'].type(self.Dtype).to(self.device)

            # print('inp.shape:',FeaDict['DegFeat']['inpfeat'].shape)
            # print('tar.shape:',FeaDict['TarSpec']['rcnspec'].shape)
            # print('opt.shape:',outputs.shape)
            # print('tag.shape:',tarFea.shape)

            loss = self.MSELoss(outputs.type(self.Dtype), tarFea.type(self.Dtype))
            # print('loss:',loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.TrLoss += loss.item()

            pbar.set_postfix_str('loss={:^7.3f}'.format(self.TrLoss/(itrNum + 1)))

        pbar.close()
        self.TrLoss /= len(self.dtaloader['tr'])

    def _val_epoch(self, Curr_epoch):
        self.VaLoss = 0

        self.model.eval()

        pbar = tqdm(self.dtaloader['va'], bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Val')

        for itrNum, FeaDict in enumerate(pbar):

            with torch.no_grad():
                outputs, output_lengths = self.model(FeaDict['DegFeat']['inpfeat'].type(self.Dtype).to(self.device), torch.LongTensor([self.sequence_length]))
                tarFea = TarFea = FeaDict['TarSpec']['rcnspec'].type(self.Dtype).to(self.device)
                loss = self.MSELoss(outputs.type(self.Dtype), tarFea.type(self.Dtype))

            self.VaLoss += loss.item()

            pbar.set_postfix_str('loss={:^7.3f};'.format(self.VaLoss/(itrNum + 1)))

        pbar.close()
        self.VaLoss /= len(self.dtaloader['va'])

    def eval(self, data):
        self.model.eval()
        data = data.to(self.device)

        # check_val = False
        # for step in range(int(data.shape[1]/self.sequence_length)+1):
        #     outputs, output_lengths = self.model(data, torch.LongTensor([self.sequence_length]))

        #     inp_data = data[:, step*300:(step+1)*300, :]
        #     # print('inp.shape:',inp_data.shape)
        #     if inp_data.shape[1] == 300 :
        #         outputs, output_lengths = self.model(inp_data, torch.LongTensor([self.sequence_length]))
        #         if step == 0 :
        #             opt_data = outputs

        #         else :
        #             opt_data = torch.cat((opt_data, outputs), 1)
        #         check_val = True
        # if check_val != True :
        #     opt_data = None

        # return opt_data


        for step in range(int(data.shape[1]/self.sequence_length)+1):
            outputs, output_lengths = self.model(data, torch.LongTensor([self.sequence_length]))

            inp_data = data[:, step*300:(step+1)*300, :]
            # print('inp.shape:',inp_data.shape)
             # 如果输入数据的长度小于 300
            if inp_data.shape[1] < 300:
                # 创建一个形状为 (1, 300, 257) 的全零张量
                padded_data = torch.zeros((1, 300, 257))

                # 将原始数据复制到新的全零张量中
                padded_data[:, :inp_data.shape[1], :] = inp_data
                outputs, output_lengths = self.model(padded_data.to(self.device), torch.LongTensor([self.sequence_length]))
                outputs = outputs[:, :inp_data.shape[1], :]
            else:
                # 如果输入数据的长度大于等于 300，则不进行填充
                padded_data = inp_data
                outputs, output_lengths = self.model(padded_data.to(self.device), torch.LongTensor([self.sequence_length]))

            # print('inp_data.shape:',inp_data.shape)
            # print('padded_data.shape:',padded_data.shape)
            # print('outputs.shape:',outputs.shape)            
            if step == 0 :
                opt_data = outputs
            else :
                opt_data = torch.cat((opt_data, outputs), 1)

        # print('opt_data.shape:',opt_data.shape)
        return opt_data

        # return self.model(data[:, i*300:(i+1)*300, :], torch.LongTensor(data.shape[self.sequence_length]))

    def _save_Mdl(self, Curr_epoch):
        print(f'Saving model to {self.MdlPth}')
        state_dict = {
            'epoch': Curr_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.VaLoss,
            }

        torch.save(state_dict, self.MdlPth+str(Curr_epoch)+'mdl.pkl')

    def _load_Mdl(self):
        checkpoint = torch.load(self.MdlPth)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.type(self.Dtype).to(self.device)