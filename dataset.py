import torch
from torch.utils.data import Dataset
import pandas as pd

class RTB(Dataset):
    def __init__(self, impressions_file_path, click_file_path, conversion_file_path):
        
        self.column_names = [
        "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP", "Region", "City",
        "Adexchange", "Domain", "URL", "AnonymousURLID", "AdslotID", "Adslotwidth",
        "Adslotheight", "Adslotvisibility", "Adslotformat", "Adslotfloorprice",
        "CreativeID", "Biddingprice", "Payingprice", "KeypageURL", "AdvertiserID", "User_tag"
        ]
        
        self.imp = pd.read_csv(impressions_file_path, delimiter='\t',names=self.column_names ,low_memory=True)
        conv = pd.read_csv(conversion_file_path, delimiter='\t',names=self.column_names, low_memory=True)
        clk = pd.read_csv(click_file_path, delimiter='\t',names=self.column_names, low_memory=True)
        
        self.clk_label = self.imp['BidID'].isin(clk['BidID'])
        self.conv_label = self.imp['BidID'].isin(conv['BidID'])

        with open('Adobe Devcraft PS/user.profile.tags.txt') as f:
            tag_dict = {}
            for idx, line in enumerate(f.readlines()):
                tag_dict[line[:5]] = line[6:]
        f.close()
        
        self.imp['User_tag'] = self.imp['User_tag'].str.split(',')
        self.imp = self.imp.explode('User_tag')
        self.imp = pd.get_dummies(self.imp, columns=['User_tag'])
        self.imp = self.imp.groupby('BidID', as_index=False).max()
        
        for key in tag_dict:
            flag = 0
            for col in self.imp.column_names:
                if col[:4] == 'User' and col[-5:] == key:
                    flag = 1
                    break
            if not flag:
                print(f"Key {key} not in column ")
                print(f"Attribute : {tag_dict[key]}")            
                self.imp[f'User_tag_{key}'] = False
        
        # for key in tag_dict:
        #      if(tag_dict[key][:-1] not in self.column_names):
        #         print(tag_dict[key][:-1])
        #         imp[tag_dict[key][:-1]] = False
                
        self.imp = self.imp.drop(['BidID', 'Logtype', 'VisitorID', 'User-Agent', 'IP', 
                        'Adexchange', 'Domain', 'URL', 'AnonymousURLID', 'AdslotID', 
                        'Adslotfloorprice', 'Biddingprice', 'KeypageURL', 'Timestamp'], axis=1)
        
        
        self.bidId = None # Not used
        self.timestamp = None # Not used
        self.visitorId = None # Not used
        self.userAgent = None # Not used
        self.ipAddress = None # Not used
        self.adSlotFloorPrice = None # Not used
        self.adExchange = None # Not used
        self.domain = None # Not used
        self.url = None # Not used
        self.anonymousURLID = None # Not used
        self.adSlotID = None # Not used
        
        self.region = None 
        self.city = None 
        self.adSlotWidth = None
        self.adSlotHeight = None
        self.adSlotVisibility = None
        self.adSlotFormat = None
        self.creativeID = None
        self.advertiserId = None
        self.userTags = None

        
    def __len__(self):
        return self.imp.shape[0]
    
    def __getitem__(self, idx):
        return self.imp.iloc[idx], self.clk_label[idx], self.conv_label[idx]
    
if __name__=='__main__':
    imp_file_path = "Adobe_dataset/imp.06.txt"
    clk_file_path = "Adobe_dataset/clk.06.txt"
    conv_file_path = "Adobe_dataset/conv.06.txt"
    dataset = RTB(imp_file_path,clk_file_path,conv_file_path)
    for i,data in enumerate(dataset):
        print(len(data))