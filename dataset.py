import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

class RTB_old(Dataset):
    def __init__(self, impressions_file_path, click_file_path, conversion_file_path, split='train'):
        
        self.split = split
        
        # Assigning Column Names to the data from ReadMe
        self.column_names = [
        "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP", "Region", "City",
        "Adexchange", "Domain", "URL", "AnonymousURLID", "AdslotID", "Adslotwidth",
        "Adslotheight", "Adslotvisibility", "Adslotformat", "Adslotfloorprice",
        "CreativeID", "Biddingprice", "Payingprice", "KeypageURL", "AdvertiserID", "User_tag"
        ]
        
        # Assigning Creative ID to a numerical Value
        self.createive_id_mapping = {'e1af08818a6cd6bbba118bb54a651961': 0, '44966cc8da1ed40c95d59e863c8c75f0': 1, '832b91d59d0cb5731431653204a76c0e': 2, '59f065a795a663140e36eec106464524': 3, '48f2e9ba15708c0146bda5e1dd653caa': 4, 'a499988a822facd86dd0e8e4ffef8532': 5, '4ad7e35171a3d8de73bb862791575f2e': 6, 'b90c12ed2bd7950c6027bf9c6937c48a': 7, '47905feeb59223468fb898b3c9ac024d': 8, '00fccc64a1ee2809348509b7ac2a97a5': 9, 'fe222c13e927077ad3ea087a92c0935c': 10, 'f65c8bdb41e9015970bac52baa813239': 11, '8dff45ed862a740986dbe688aafee7e5': 12, '4b724cd63dfb905ebcd54e64572c646d': 13, 'e049ebe262e20bed5f9b975208db375b': 14, 'cb7c76e7784031272e37af8e7e9b062c': 15, '612599432d200b093719dd1f372f7a30': 16, '23485fcd23122d755d38f8c89d46ca56': 17, '0cd33fcb336655841d3e1441b915748d': 18, '011c1a3d4d3f089a54f9b70a4c0a6eb3': 19, '13606a7c541dcd9ca1948875a760bb31': 20, 'd881a6c788e76c2c27ed1ef04f119544': 21, '80a776343079ed94d424f4607b35fd39': 22, 'd5cecca9a6cbd7a0a48110f1306b26d1': 23, '77819d3e0b3467fe5c7b16d68ad923a1': 24, '2f88fc9cf0141b5bbaf251cab07f4ce7': 25, '86c2543527c86a893d4d4f68810a0416': 26, '3d8f1161832704a1a34e1ccdda11a81e': 27, 'd01411218cc79bc49d2a4078c4093b76': 28, '2abc9eaf57d17a96195af3f63c45dc72': 29, '6b9331e0f0dbbfef42c308333681f0a3': 30, '7eb0065067225fa5f511f7ffb9895f24': 31, '23d6dade7ed21cea308205b37594003e': 32, 'c936045d792f6ea3aa22d86d93f5cf23': 33, 'fb5afa9dba1274beaf3dad86baf97e89': 34, '4400bf8dea968a0068824792fd336c4c': 35, '7097e4210dea4d69f07f0f5e4343529c': 36, '1a43f1ff53f48573803d4a3c31ebc163': 37, '82f125e356439d73902ae85e2be96777': 38, '5c4e0bb0db45e2d1b3a14f817196ebd6': 39, 'ff5123fb9333ca095034c62fdaaf51aa': 40, '62f7f9a6dca2f80cc00f17dcda730bc1': 41, 'c938195f9e404b4f38c7e71bf50263e5': 42, '3b805a00d99d5ee2493c8fb0063e30e9': 43, '87945ed58e806dbdc291b3662f581354': 44, 'e87d7633d474589c2e2e3ba4eda53f6c': 45, '6cdf8fdd3e01122b09b5b411510a2385': 46, '0055e8503dc053435b3599fe44af118b': 47, 'bc27493ad2351e2577bc8664172544f8': 48}
        
        # Reading Data (Reading impression data as it has User_Tag which is neccessary for training)
        self.imp = pd.read_csv(impressions_file_path, delimiter='\t',names=self.column_names ,low_memory=True)
        conv = pd.read_csv(conversion_file_path, delimiter='\t',names=self.column_names, low_memory=True)
        clk = pd.read_csv(click_file_path, delimiter='\t',names=self.column_names, low_memory=True)
        
        # Mapping Creative ID to a numerical Value
        self.imp['CreativeID'] = self.imp['CreativeID'].map(self.createive_id_mapping)
        
        # Reading User Profile
        with open('Adobe Devcraft PS/user.profile.tags.txt') as f:
            tag_dict = {}
            for idx, line in enumerate(f.readlines()):
                tag_dict[line[:5]] = line[6:]
        f.close()
        
        # One hot Encoding User Tags
        self.imp['User_tag'] = self.imp['User_tag'].str.split(',')
        self.imp = self.imp.explode('User_tag')
        self.imp = pd.get_dummies(self.imp, columns=['User_tag'])
        user_cols = [col for col in self.imp.columns if col.startswith('User_')]
        self.imp = self.imp.groupby('BidID', as_index=False).agg({**{col: 'max' for col in user_cols}, **{col: 'first' for col in self.imp.columns if col not in user_cols + ['BidID']}})
        
        # Adding User Tag column which is not present in all the samples
        for key in tag_dict:
            flag = 0
            for col in self.imp.columns:
                if col[:4] == 'User' and col[-5:] == key:
                    flag = 1
                    break
            if not flag:
                print(f"Key {key} not in column ")
                print(f"Attribute : {tag_dict[key]}")            
                self.imp[f'User_tag_{key}'] = False
        
        self.clk_label = self.imp['BidID'].isin(clk['BidID'])
        self.conv_label = self.imp['BidID'].isin(conv['BidID'])
        
        # Dropping Unneccessary Columns
        self.imp = self.imp.drop(['BidID', 'Logtype', 'VisitorID', 'User-Agent', 'IP', 
                        'Adexchange', 'Domain', 'URL', 'AnonymousURLID', 'AdslotID', 
                        'Adslotfloorprice', 'Biddingprice', 'KeypageURL', 'Timestamp'], axis=1)
        
        
        stratify_label = self.clk_label 
        self.imp_train, self.imp_val, self.clk_train, self.clk_val, self.conv_train, self.conv_val = train_test_split(
            self.imp, self.clk_label, self.conv_label, train_size=0.95, stratify=stratify_label
)       
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.imp_train = torch.tensor(self.imp_train.values.astype('float32'), device=device)
        self.imp_val = torch.tensor(self.imp_val.values.astype('float32'), device=device)
        self.clk_train = torch.tensor(self.clk_train.values.astype('float32'), device=device)
        self.clk_val = torch.tensor(self.clk_val.values.astype('float32'), device=device)
        self.conv_train = torch.tensor(self.conv_train.values.astype('float32'), device=device)
        self.conv_val = torch.tensor(self.conv_val.values.astype('float32'), device=device)        
                        
        print(f"imp_train shape: {self.imp_train.shape}")
        print(f"imp_val shape: {self.imp_val.shape}")
        print(f"clk_train shape: {self.clk_train.shape}")
        print(f"clk_val shape: {self.clk_val.shape}")
        print(f"conv_train shape: {self.conv_train.shape}")
        print(f"conv_val shape: {self.conv_val.shape}") 
        
    def __len__(self):
        return len(self.imp_train) if self.split == "train" else len(self.imp_val)

    def __getitem__(self, idx):
        if self.split == "train":
            return self.imp_train[idx], self.clk_train[idx], self.conv_train[idx]
        else:
            return self.imp_val[idx], self.clk_val[idx], self.conv_val[idx]
    
class RTB(Dataset):
    def __init__(self, impressions_file_path, click_file_path, conversion_file_path, split="train"):
        self.split = split  # Can be 'train' or 'val'
        
        # Column names from ReadMe
        self.column_names = [
        "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP", "Region", "City",
        "Adexchange", "Domain", "URL", "AnonymousURLID", "AdslotID", "Adslotwidth",
        "Adslotheight", "Adslotvisibility", "Adslotformat", "Adslotfloorprice",
        "CreativeID", "Biddingprice", "Payingprice", "KeypageURL", "AdvertiserID", "User_tag"
        ]
        
        # Mapping CreativeID to numerical values
        self.creative_id_mapping = {'e1af08818a6cd6bbba118bb54a651961': 0, '44966cc8da1ed40c95d59e863c8c75f0': 1, '832b91d59d0cb5731431653204a76c0e': 2, '59f065a795a663140e36eec106464524': 3, '48f2e9ba15708c0146bda5e1dd653caa': 4, 'a499988a822facd86dd0e8e4ffef8532': 5, '4ad7e35171a3d8de73bb862791575f2e': 6, 'b90c12ed2bd7950c6027bf9c6937c48a': 7, '47905feeb59223468fb898b3c9ac024d': 8, '00fccc64a1ee2809348509b7ac2a97a5': 9, 'fe222c13e927077ad3ea087a92c0935c': 10, 'f65c8bdb41e9015970bac52baa813239': 11, '8dff45ed862a740986dbe688aafee7e5': 12, '4b724cd63dfb905ebcd54e64572c646d': 13, 'e049ebe262e20bed5f9b975208db375b': 14, 'cb7c76e7784031272e37af8e7e9b062c': 15, '612599432d200b093719dd1f372f7a30': 16, '23485fcd23122d755d38f8c89d46ca56': 17, '0cd33fcb336655841d3e1441b915748d': 18, '011c1a3d4d3f089a54f9b70a4c0a6eb3': 19, '13606a7c541dcd9ca1948875a760bb31': 20, 'd881a6c788e76c2c27ed1ef04f119544': 21, '80a776343079ed94d424f4607b35fd39': 22, 'd5cecca9a6cbd7a0a48110f1306b26d1': 23, '77819d3e0b3467fe5c7b16d68ad923a1': 24, '2f88fc9cf0141b5bbaf251cab07f4ce7': 25, '86c2543527c86a893d4d4f68810a0416': 26, '3d8f1161832704a1a34e1ccdda11a81e': 27, 'd01411218cc79bc49d2a4078c4093b76': 28, '2abc9eaf57d17a96195af3f63c45dc72': 29, '6b9331e0f0dbbfef42c308333681f0a3': 30, '7eb0065067225fa5f511f7ffb9895f24': 31, '23d6dade7ed21cea308205b37594003e': 32, 'c936045d792f6ea3aa22d86d93f5cf23': 33, 'fb5afa9dba1274beaf3dad86baf97e89': 34, '4400bf8dea968a0068824792fd336c4c': 35, '7097e4210dea4d69f07f0f5e4343529c': 36, '1a43f1ff53f48573803d4a3c31ebc163': 37, '82f125e356439d73902ae85e2be96777': 38, '5c4e0bb0db45e2d1b3a14f817196ebd6': 39, 'ff5123fb9333ca095034c62fdaaf51aa': 40, '62f7f9a6dca2f80cc00f17dcda730bc1': 41, 'c938195f9e404b4f38c7e71bf50263e5': 42, '3b805a00d99d5ee2493c8fb0063e30e9': 43, '87945ed58e806dbdc291b3662f581354': 44, 'e87d7633d474589c2e2e3ba4eda53f6c': 45, '6cdf8fdd3e01122b09b5b411510a2385': 46, '0055e8503dc053435b3599fe44af118b': 47, 'bc27493ad2351e2577bc8664172544f8': 48}

        # Loading datasets
        self.imp = pd.read_csv(impressions_file_path, delimiter='\t',names=self.column_names ,low_memory=True)
        conv = pd.read_csv(conversion_file_path, delimiter='\t',names=self.column_names, low_memory=True)
        clk = pd.read_csv(click_file_path, delimiter='\t',names=self.column_names, low_memory=True)

        # Convert CreativeID to numerical values
        self.imp['CreativeID'] = self.imp['CreativeID'].map(self.creative_id_mapping)

        # One-hot encode user tags
        self.imp['User_tag'] = self.imp['User_tag'].str.split(',')
        self.imp = self.imp.explode('User_tag')
        self.imp = pd.get_dummies(self.imp, columns=['User_tag'])

        user_cols = [col for col in self.imp.columns if col.startswith('User_')]
        self.imp = self.imp.groupby('BidID', as_index=False).agg({
            **{col: 'max' for col in user_cols}, 
            **{col: 'first' for col in self.imp.columns if col not in user_cols + ['BidID']}
        })

        # Labels for clicks and conversions
        self.clk_label = self.imp['BidID'].isin(clk['BidID']).astype(float)
        self.conv_label = self.imp['BidID'].isin(conv['BidID']).astype(float)

        # Drop unnecessary columns
        self.imp = self.imp.drop(['BidID', 'Logtype', 'VisitorID', 'User-Agent', 'IP', 
                        'Adexchange', 'Domain', 'URL', 'AnonymousURLID', 'AdslotID', 
                        'Adslotfloorprice', 'Biddingprice', 'KeypageURL', 'Timestamp'], axis=1)

        # Train-validation split
        stratify_label = self.clk_label
        self.imp_train, self.imp_val, self.clk_train, self.clk_val, self.conv_train, self.conv_val = train_test_split(
            self.imp, self.clk_label, self.conv_label, train_size=0.95, stratify=stratify_label
        )

        # Convert to PyTorch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.imp_train = torch.tensor(self.imp_train.values.astype('float32'), device=device)
        self.imp_val = torch.tensor(self.imp_val.values.astype('float32'), device=device)
        self.clk_train = torch.tensor(self.clk_train.values.astype('float32'), device=device)
        self.clk_val = torch.tensor(self.clk_val.values.astype('float32'), device=device)
        self.conv_train = torch.tensor(self.conv_train.values.astype('float32'), device=device)
        self.conv_val = torch.tensor(self.conv_val.values.astype('float32'), device=device)        

        del conv,clk
        del stratify_label
        del self.column_names
        del self.creative_id_mapping
        del self.imp
        del self.clk_label
        del self.conv_label

        print(f"Train dataset size: {self.imp_train.shape}")
        print(f"Validation dataset size: {self.imp_val.shape}")

    def __len__(self):
        return len(self.imp_train) if self.split == "train" else len(self.imp_val)

    def __getitem__(self, idx):
        if self.split == "train":
            return self.imp_train[idx], self.clk_train[idx], self.conv_train[idx]
        else:
            return self.imp_val[idx], self.clk_val[idx], self.conv_val[idx]
    
if __name__=='__main__':
    imp_file_path = "Adobe_dataset/imp.06.txt"
    clk_file_path = "Adobe_dataset/clk.06.txt"
    conv_file_path = "Adobe_dataset/conv.06.txt"
    # dataset = RTB(imp_file_path,clk_file_path,conv_file_path, split='train')
    dataset = RTB_old(imp_file_path,clk_file_path,conv_file_path)
    dataloader = DataLoader(dataset=dataset, batch_size=2)
    for i,data in enumerate(dataloader):
        for j in range(len(data)):
            print(data[j].shape)
        break