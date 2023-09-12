import torch.nn as nn

class MultiLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers, output_size):
        '''
        __init__ function is to define the LSTM model
        :param input_size: The number of input variables
        :param hidden_size: The number of LSTM units in each layer
        :param num_layers: The number of LSTM layers
        :param output_size: The number of output variables

        '''
        super(MultiLSTM,self).__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        '''
        forward function is to perform the forward pass through the LSTM model
        :param x:  The input vector
        :return:
        '''
        out,_=self.lstm(x)
        out=self.fc(out[:,:])
        return out