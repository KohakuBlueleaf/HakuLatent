"""
A demo script to finetune a pretrained VAE on ImageNet with the EQ-VAE setup.
"""

if __name__ == "__main__":
    import torch
    import torch.utils.data as data
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from torchvision.transforms import (
        Compose,
        Resize,
        RandomCrop,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        ToTensor,
        Lambda,
    )
    from diffusers import AutoencoderKL
    from convnext_perceptual_loss import ConvNextType

    from hl_dataset.imagenet import ImageNetDataset
    from hakulatent.transform import (
        LatentTransformCompose,
        LatentTransformSwitch,
        RotationTransform,
        ScaleDownTransform,
        ScaleUpCropTransform,
        CropTransform,
        RandomAffineTransform,
        BlendingTransform,
    )
    from hakulatent.trainer import LatentTrainer
    from hakulatent.losses import AdvLoss, ReconLoss, KeplerQuantizerRegLoss
    from hakulatent.extune.linear import ScaleLinear, ScaleConv2d
else:
    # This if-else can speedup multi-worker dataloader in windows
    print("Subprocess Starting:", __name__)


BASE_MODEL = "KBlueLeaf/EQ-SDXL-VAE"
SUB_FOLDER = None
EPOCHS = 1
BATCH_SIZE = 8
GRAD_ACC = 8
GRAD_CKPT = False
TRAIN_DEC_ONLY = True

LOSS_TYPE = "mse"
LPIPS_NET = "vgg"
USE_CONVNEXT = True
ADV_START_ITER = 0

NUM_WORKERS = 8
SIZE = 256
LR = 1e-3
DLR = 1e-3

NEW_LATENT_DIM = None
PRETRAIN = False


def process(x):
    return x * 2 - 1


def deprocess(x):
    return x * 0.5 + 0.5


if __name__ == "__main__":
    split = "train"
    transform = Compose(
        [
            Resize(SIZE),
            RandomCrop((SIZE, SIZE)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            Lambda(process),
        ]
    )
    dataset = ImageNetDataset(split, transform)
    loader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=bool(NUM_WORKERS),
    )
    # loader warmup
    next(iter(loader))

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder=SUB_FOLDER)
    if GRAD_CKPT:
        vae.enable_gradient_checkpointing()

    if NEW_LATENT_DIM:
        vae.config.latent_channels = NEW_LATENT_DIM
        vae.encoder.conv_out = ScaleConv2d(
            "",
            vae.encoder.conv_out,
            vae.encoder.conv_out.in_channels,
            NEW_LATENT_DIM * 2,
        ).generate_module()
        vae.decoder.conv_in = ScaleConv2d(
            "", vae.decoder.conv_in, NEW_LATENT_DIM, vae.decoder.conv_in.out_channels
        ).generate_module()
        vae.quant_conv = ScaleConv2d(
            "", vae.quant_conv, NEW_LATENT_DIM * 2, NEW_LATENT_DIM * 2, inputs_groups=[]
        ).generate_module()
        vae.post_quant_conv = ScaleConv2d(
            "", vae.post_quant_conv, NEW_LATENT_DIM, NEW_LATENT_DIM
        ).generate_module()

    if PRETRAIN:
        vae = AutoencoderKL(
            down_block_types=["DownEncoderBlock2D"] * 4,
            block_out_channels=[128, 256, 512, 512],
            latent_channels=NEW_LATENT_DIM,
            up_block_types=["UpDecoderBlock2D"] * 4,
            layers_per_block=2,
        )
        vae.save_pretrained("./models/Kohaku-VAE")

    if TRAIN_DEC_ONLY:
        vae.requires_grad_(False)
        vae.decoder.requires_grad_(True)

    vae.get_last_layer = lambda: vae.decoder.conv_out.weight

    trainer_module = LatentTrainer(
        vae=vae,
        recon_loss=ReconLoss(
            loss_type=LOSS_TYPE,
            lpips_net=LPIPS_NET,
            convnext_type=ConvNextType.TINY if USE_CONVNEXT else None,
            convnext_kwargs={
                "feature_layers": [2, 6, 10, 14],
                "use_gram": False,
                "input_range": (-1, 1),
                "device": "cuda",
            },
            loss_weights={
                LOSS_TYPE: 0.25,
                "lpips": 0.3,
                "convnext": 0.45,
            },
        ),
        latent_loss=KeplerQuantizerRegLoss(
            embed_dim=vae.config.latent_channels,
            scale=1,
            partitions=1,
            num_embed=1024,
            beta=0.25,
            use_kepler_loss=False,
        ),
        # adv_loss=AdvLoss(start_iter=ADV_START_ITER, disc_loss="vanilla", n_layers=5),
        img_deprocess=deprocess,
        log_interval=100,
        transform_prob=0.5,
        latent_transform=LatentTransformCompose(
            RandomAffineTransform(
                rotate_range=(-180, 180),
                scale_range=(0.8, 1.2),
                shear_range=((-10, 10), (-5, 5)),
                translate_range=(0.1, 0.1),
                method="random",
            ),  # Thanks AmericanPresidentJimmyCarter
            BlendingTransform([0.1, 0.4], method="random"),
        ),
        loss_weights={
            "recon": 1 / 8,
            "adv": 0.25,
            "kl": 0,
            "reg": 1.0,
        },
        name="EQ-SDXL-VAE-random-affine",
        lr=LR,
        lr_disc=DLR,
        optimizer="torch.optim.AdamW",
        opt_configs={"betas": (0.9, 0.98), "weight_decay": 1e-2},
        lr_sch_configs={
            "lr": {
                "end": EPOCHS * (len(loader) + 1) // GRAD_ACC,
                "value": 1.0,
                "min_value": 0.1,
                "mode": "cosine",
                "warmup": 0,
            }
        },
        grad_acc=GRAD_ACC,
    )

    logger = WandbLogger(
        project="HakuLatent",
        name="EQ-VAE-ch8-randomaffine-KepRegLoss",
        # offline=True
    )
    trainer = pl.Trainer(
        logger=logger,
        devices=1,
        max_epochs=EPOCHS,
        precision="16-mixed",
        callbacks=[
            ModelCheckpoint(every_n_train_steps=1000),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=1,
    )
    trainer.fit(trainer_module, loader)
