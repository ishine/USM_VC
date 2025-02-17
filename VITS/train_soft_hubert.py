import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import librosa
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import commons
import utils
from data_utils_hubert import (
    TextAudioLoader_soft,
    TextAudioLoader_ppgmap,
    TextAudioLoader_map2phn_Gdict,
    TextAudioLoader_map2phn_Mixdict,
    TextAudioLoader_map2phn_Mixdict2,
    TextAudioLoader_map2phn_Mixdict3,
    TextAudioCollate,
    DistributedBucketSampler
)
from models_nsf import (
    SynthesizerTrn,
    SynthesizerTrnMs768NSFsid_nono,
    MultiPeriodDiscriminator,
    MultiPeriodDiscriminatorV2,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '25565'

    hps = utils.get_hparams()
    
    if n_gpus > 1:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    else:
        run(0, 1, hps)


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    if n_gpus > 1:
        dist.init_process_group(
            backend='nccl', init_method='env://', world_size=n_gpus, rank=rank,
        )
    torch.cuda.set_device(rank)
    torch.manual_seed(hps.train.seed)

    if rank == 0:
        logging.info("Prepare data loader...")
    train_dataset = TextAudioLoader_soft(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000],  # 20s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioCollate(ling_feat_dim=hps.data.ling_feat_dim)
    train_loader = DataLoader(
        train_dataset,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        eval_dataset = TextAudioLoader_soft(hps.data.validation_files, hps.data, is_train=False)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=2,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    if rank == 0:
        logging.info("Initialize model...")

    #net_g = SynthesizerTrnMs768NSFsid_nono(
    net_g = SynthesizerTrn(        
        ling_dim=hps.data.ling_feat_dim,
        pitch_type=hps.data.pitch_type,
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda(rank)
    #net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    net_d = MultiPeriodDiscriminatorV2(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(), 
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps,
    )
    if n_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank])

    if rank == 0:
        logging.info(f"Find and load checkpoint if any...")
    try:
        latest_G_ckpt = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        global_step = int("".join(filter(str.isdigit, os.path.basename(latest_G_ckpt))))
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        # global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    if rank == 0:
        logging.info(f"Start Epoch {epoch_str}")
        logging.info(f"Start global steps {global_step}")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    if rank == 0:
        logging.info(f"Begin training model...")

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], 
                [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader],
                logger, [writer, writer_eval]
            )
        else:
            train_and_evaluate(
                rank, epoch, hps, [net_g, net_d], [optim_g, optim_d],
                [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, (
        ling,
        ling_lengths,
        pitch,
        f0,
        spec,
        spec_lengths,
        wav,
        wav_lengths,
        speaker_ids,
        style_ids,
    ) in enumerate(train_loader):
        ling = ling.cuda(rank, non_blocking=True)
        ling_lengths = ling_lengths.cuda(rank, non_blocking=True)
        pitch = pitch.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        wav, wav_lengths = wav.cuda(rank, non_blocking=True), wav_lengths.cuda(rank, non_blocking=True)
        speaker_ids = speaker_ids.cuda(rank, non_blocking=True)
        style_ids = style_ids.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = net_g(ling, ling_lengths, pitch, f0, spec, spec_lengths, speaker_ids, style_ids)
            #) = net_g(ling, ling_lengths, spec, spec_lengths, speaker_ids, style_ids) 

            mel = spec_to_mel_torch(
                spec, 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate,
                hps.data.mel_fmin, 
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate, 
                hps.data.hop_length, 
                hps.data.win_length, 
                hps.data.mel_fmin, 
                hps.data.mel_fmax
            )

            wav = commons.slice_segments(
                wav, ids_slice * hps.data.hop_length, hps.train.segment_size
            ) # slice 

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wav, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wav, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g) * 20
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                logger.info(
                    'Train Epoch: {} [{:.0f}%]'.format(
                        epoch, 100. * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])
            
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl})

                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = { 
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )
            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                ck_G_list = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth", latest=False)
                ck_num = len(ck_G_list)
                if ck_num > 10:
                    for old_ckpt in ck_G_list[0:ck_num-10]:
                        utils.remove_file(old_ckpt)
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                ck_D_list = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth",latest=False)
                if ck_num > 10:
                    for old_ckpt in ck_D_list[0:ck_num-10]:
                        utils.remove_file(old_ckpt)               
        global_step += 1
      
    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))
 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (
            ling,
            ling_lengths,
            pitch,
            f0,
            spec,
            spec_lengths,
            wav,
            wav_lengths,
            speaker_ids,
            style_ids,
        ) in enumerate(eval_loader):
            ling, ling_lengths = ling.cuda(0), ling_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            wav,  wav_lengths  = wav.cuda(0),  wav_lengths.cuda(0)
            speaker_ids = speaker_ids.cuda(0)
            style_ids = style_ids.cuda(0)
            pitch = pitch.cuda(0)
            f0 = f0.cuda(0)
            speaker_ids = speaker_ids.cuda(0)
            style_ids = style_ids.cuda(0)
            # remove else
            ling = ling[:1]
            ling_lengths = ling_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            wav = wav[:1]
            wav_lengths = wav_lengths[:1]
            speaker_ids = speaker_ids[:1]
            style_ids = style_ids[:1]
            pitch = pitch[:1]
            f0 = f0[:1]
            break
        if hasattr(generator, 'module'):
            y_hat, mask, *_ = generator.module.infer(
                ling, ling_lengths, pitch, f0, speaker_ids, style_ids, max_len=1000
            )
        else:
            y_hat, mask, *_ = generator.infer(
                ling, ling_lengths, pitch, f0, speaker_ids, style_ids, max_len=1000
            )
        y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

        mel = spec_to_mel_torch(
            spec, 
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate,
            hps.data.mel_fmin, 
            hps.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
        image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": wav[0,:,:wav_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step, 
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()
