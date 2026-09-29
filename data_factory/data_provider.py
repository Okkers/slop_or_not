from data_loader import Dataset_MTT_Detection
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = None # Set for real data

    if flag == 'test':
        ## TO-DO
        pass 

    elif flag == 'pred':
        ## TO-DO 
        pass 

    else:
        ## TO-DO, train or validation
        pass 

    data_set = None # Set for loading dataset

    print(flag, len(data_set))
    data_loader = None # Set for loading data loader

    return data_set, data_loader