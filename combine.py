# This file combines the impressions csv files
# and adds the conv_label and clk_label columns as well
# as separates the user_tags into separate columns
import pandas as pd

with open('Adobe Devcraft PS/user.profile.tags.txt') as f:
    tag_dict = {}
    for idx, line in enumerate(f.readlines()):
        tag_dict[line[:5]] = line[6:]

file_nums = ['06', '07', '08', '09', '10', '11', '12']
column_names = [
    "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP", "Region", "City",
    "Adexchange", "Domain", "URL", "AnonymousURLID", "AdslotID", "Adslotwidth",
    "Adslotheight", "Adslotvisibility", "Adslotformat", "Adslotfloorprice",
    "CreativeID", "Biddingprice", "Payingprice", "KeypageURL", "AdvertiserID", "User_tag"
]

combined_imp = pd.DataFrame(columns=column_names)
combined_clk = pd.DataFrame(columns=column_names)
combined_conv = pd.DataFrame(columns=column_names)

for i in range(7):
    imp = pd.read_csv("dataset/imp."+file_nums[i]+".txt", delimiter='\t', names=column_names, low_memory=True)
    clk = pd.read_csv("dataset/clk."+file_nums[i]+".txt", delimiter='\t', names=column_names, low_memory=True)
    conv = pd.read_csv("dataset/conv."+file_nums[i]+".txt", delimiter='\t', names=column_names, low_memory=True)

    combined_imp = pd.concat([combined_imp, imp], ignore_index=True)
    combined_clk = pd.concat([combined_clk, clk], ignore_index=True)
    combined_conv = pd.concat([combined_conv, conv], ignore_index=True)

print(combined_imp.shape)
print(combined_clk.shape)
print(combined_conv.shape)

combined_imp.to_csv('dataset_combined/imp.csv', index=False)
combined_clk.to_csv('dataset_combined/clk.csv', index=False)
combined_conv.to_csv('dataset_combined/conv.csv', index=False)