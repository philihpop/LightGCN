import world
import dataloader
import model
import utils
from pprint import pprint
from pruned_dataloader import PrunedDataset
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'beauty':  
    dataset = dataloader.AmazonBeauty(path="../data/beauty")
elif world.dataset == 'pruned':  
    dataset = PrunedDataset(data_dir="../data/beauty/pruned_data")
print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}