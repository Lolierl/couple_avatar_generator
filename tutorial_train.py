from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",                
    filename="controlLDM-step{step}",     
    every_n_train_steps=1000,             
    save_top_k=-1,                        
    save_last=True                        
)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=2, precision=16, callbacks=[logger, checkpoint_callback], strategy='ddp', accumulate_grad_batches=4)

# Train!
trainer.fit(model, dataloader)
