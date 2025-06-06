import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 5 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()
    if world.dataset == 'pruned':
        cprint("[GENERATING SUBMISSION]")
        try:
            submission_df = dataset.get_submission_recommendations(
                Recmodel, 
                top_k=10, 
                output_file='submission.csv'
            )
            print("Submission file 'submission.csv' generated.")
            print(f"Generated recommendations for {len(submission_df)} users")
        except Exception as e:
            print(f"Error generating submission: {e}")
            print("Make sure you ran the setup script first to create the pruned dataset")
    
    # For original datasets, use the traditional method
    elif hasattr(dataset, 'get_submission_recommendations'):
        cprint("[GENERATING SUBMISSION]")
        try:
            submission_df = dataset.get_submission_recommendations(
                Recmodel,
                submission_file='../data/beauty/sample_submission.csv',
                top_k=10
            )
            submission_df.to_csv('submission.csv', index=False)
            print("Submission file 'submission.csv' generated.")
        except Exception as e:
            print(f"Error generating submission: {e}")
    
    print("\n Training and submission generation completed.")