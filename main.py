import glob
import pandas as pd
import pickle
from BertModel import BERT
import torch
import os
from variables import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

model = BERT()

#checkpoint_callback = ModelCheckpoint(
#    filepath=os.getcwd(),
#    save_best_only=False,
#    verbose=True,
#    monitor='loss',
#    mode='min',
#    prefix=''
#)

#early_stop_callback = EarlyStopping(
#    monitor='loss',
#    min_delta=0.00,
#    patience=PATIENCE,
#    verbose=True,
#    mode='min'
#)

#trainer = Trainer(gpus=1, \
#                  show_progress_bar=True, \
#                  max_nb_epochs=MAX_NB_EPOCHS, \
#                  early_stop_callback=early_stop_callback)

trainer = Trainer(gpus=1, show_progress_bar=True, max_epochs=5, early_stop_callback=False)

trainer.fit(model)
torch.save(model.state_dict(), "bert-initial-weights.pt")
