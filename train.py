from share import *
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = '/root/autodl-tmp/ControlNet/models/control_sd15_ini.ckpt'
checkpoint_path = '/root/autodl-tmp/ControlNet/checkpoints/last.ckpt'
batch_size = 1
logger_freq = 2000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset(dataset_path='test')
print("Dataset size:", len(dataset))
dataloader = DataLoader(dataset, num_workers=16, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="{epoch:02d}",
    save_top_k=0,  
    save_last=True, 
    every_n_train_steps=1000
)

trainer = pl.Trainer(gpus=4, precision=16, callbacks=[logger,  checkpoint_callback], strategy='ddp', accumulate_grad_batches=16, max_epochs=500, enable_checkpointing=True, resume_from_checkpoint=checkpoint_path if os.path.exists(checkpoint_path) else None)

# Train!
trainer.fit(model, dataloader)
