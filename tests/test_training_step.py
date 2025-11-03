import torch
from torch.utils.data import DataLoader
from src.lightgan_ld.registry import build_models
from src.lightgan_ld.data.dummy import DummyPairs
from src.lightgan_ld.engine.trainer import TrainObjects, make_optimizers, train_step
from src.lightgan_ld.losses.perceptual_vgg import VGGPerceptual
from src.lightgan_ld.utils.metrics import LPIPSMetric
from src.lightgan_ld.utils.logging import make_tb_logger

def minimal_cfg():
    return {
      "device": "cpu",
      "train": {"amp": False, "optimizer":{"lr":1e-3,"betas":[0.5,0.999],"weight_decay":0.0},
                "onecycle":{"enabled":False},"print_every":1,"val_every":10,
                "swa":{"enabled":False},"log_dir":"./runs"},
      "loss":{"adv_weight":1.0,"perceptual_weight":0.0,"ssim_weight":0.0,"ffl_weight":0.0,"edge_weight":0.0},
      "_num_train_steps_per_epoch": 1,
      "model":{"sinogram_encoder":{"enabled":False,"penc":0.0,"channels":8},
               "generator":{"in_channels":1,"base_channels":16,"num_down":2,"use_ghost":True,"use_condconv":False,"use_eca":False,"condconv_experts":2},
               "discriminator":{"in_channels":1,"base_channels":8,"spectral_norm":False}}
    }

def test_single_train_step():
    cfg = minimal_cfg()
    from src.lightgan_ld.models.generator import LightGANLDGenerator
    from src.lightgan_ld.models.discriminator import PatchDiscriminator
    from src.lightgan_ld.models.sinogram_encoder import SinogramEncoder
    G = LightGANLDGenerator(in_ch=1, base=16, num_down=2, use_ghost=True, use_condconv=False, use_eca=False, experts=2)
    D = PatchDiscriminator(in_ch=1, base=8, spectral_norm=False)
    E = SinogramEncoder(in_ch=1, out_ch=1, channels=8)
    opt_G,opt_D,sch_G,sch_D = make_optimizers(G,D,{"train":cfg["train"],"_num_train_steps_per_epoch":1})
    objs = TrainObjects(G,D,E,opt_G,opt_D,sch_G,sch_D,torch.cuda.amp.GradScaler(enabled=False),None,None,VGGPerceptual(),LPIPSMetric())
    ds = DummyPairs(n=2, img_size=64)
    loader = DataLoader(ds, batch_size=2)
    tb = make_tb_logger(cfg["train"]["log_dir"])
    for batch in loader:
        out = train_step(batch, objs, cfg, 0, tb)
        assert "d_loss" in out and "g_loss" in out
        break
