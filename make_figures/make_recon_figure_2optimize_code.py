# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.utils.data
from torchvision.utils import save_image
import random
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
import lpips_loss
import tqdm
import pickle
from PIL import Image

lreq.use_implicit_lreq.set(True)


def place(canvas, image, x, y):
    im_size = image.shape[2]
    if len(image.shape) == 4:
        image = image[0]
    canvas[:, y: y + im_size, x: x + im_size] = image * 0.5 + 0.5


def save_sample(model, sample, i):
    os.makedirs('results', exist_ok=True)

    with torch.no_grad():
        model.eval()
        x_rec = model.generate(model.generator.layer_count - 1, 1, z=sample)

        def save_pic(x_rec):
            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample,
                       'sample_%i_lr.png' % i, nrow=16)

        save_pic(x_rec)


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)
    model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return Z

    def decode(x, noise):
        layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=noise)

    path = cfg.DATASET.SAMPLES_PATH
    im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)

    paths = list(os.listdir(path))

    paths = sorted(paths)
    random.seed(1)
    random.shuffle(paths)

    def get_initial_latent_code(filename):
        with torch.no_grad():
            img = np.asarray(Image.open(path + '/' + filename))
            if img.shape[2] == 4:
                img = img[:, :, :3]
            im = img.transpose((2, 0, 1))
            x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
            if x.shape[0] == 4:
                x = x[:3]
            factor = x.shape[2] // im_size
            if factor != 1:
                x = torch.nn.functional.avg_pool2d(x[None, ...], factor, factor)[0]
            assert x.shape[2] == im_size
            latents = encode(x[None, ...].cuda())
        return x, latents

    def latent2input(latent):
        return latent.unsqueeze(1).expand(-1, model.mapping_fl.num_layers, -1)

    percept = lpips_loss.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=True
    )

    def optimize(x, latent, mse_coef=10, initial_lr=0.1, step=1000, lr_rampdown=0.25, lr_rampup=0.05):
        def get_lr(t, initial_lr, rampdown=lr_rampdown, rampup=lr_rampup):
            lr_ramp = min(1, (1 - t) / rampdown)
            lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
            lr_ramp = lr_ramp * min(1, t / rampup)

            return initial_lr * lr_ramp

        def tensor_reshape(img):
            batch, channel, height, width = img.shape
            if height > 256:
                factor = height // 256

                img = img.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img = img.mean([3, 5])
            return img

        # def latent_noise(latent, strength):
        #     noise = torch.randn_like(latent) * strength
        #
        #     return latent + noise
        batch_size = x.shape[0]
        x_resized = tensor_reshape(x)
        latent.requires_grad = True

        #noises = []
        #for res in model.decoder.layer_to_resolution:
        #    for _ in range(2):
        #        noise = torch.randn(batch_size, 1, res, res).cuda()
        #        noises.append(noise)
        noises = True

        optimizer = torch.optim.Adam([latent], lr=initial_lr)
        pbar = tqdm.tqdm(range(step))

        for i in pbar:
            t = i / step
            lr = get_lr(t, initial_lr, rampdown=lr_rampdown, rampup=lr_rampup)
            optimizer.param_groups[0]['lr'] = lr

            # noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            # latent_n = latent_noise(latent_in, noise_strength.item())

            output = model.decoder(latent2input(latent), layer_count - 1, 1, noise=noises)
            output_resized = tensor_reshape(output)

            percept_loss = percept(output_resized, x_resized).flatten().sum()
            mse_loss = F.mse_loss(output, x, reduction='none'
                                  ).view(batch_size, -1).mean(dim=-1).sum()

            loss = percept_loss + mse_coef * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (
                    f'perceptual: {percept_loss.item() / batch_size:.4f};'
                    f' mse: {mse_loss.item() / batch_size:.4f}; lr: {lr:.4f}'
                )
            )

        return latent  # , noises

    def make(paths, batch_size=2):
        canvas = []
        batch_start = 0
        while batch_start < len(paths):
            batch_end = batch_start + batch_size

            xs, latents = [], []
            for filename in paths[batch_start:batch_end]:
                x, lat = get_initial_latent_code(filename)
                lat = lat[:, 0]
                xs.append(x)
                latents.append(lat)
            xs = torch.stack(xs)
            latents = torch.cat(latents, dim=0)

            #latents_new, noises = optimize(xs.detach(), latents.clone())
            latents_new = optimize(xs.detach(), latents.clone())
            noises = True

            f = decode(latent2input(latents), noises)
            f_new = decode(latent2input(latents_new), noises)
            r = torch.cat([xs.detach().cpu(), f.detach().cpu(), f_new.detach().cpu()], dim=3)
            for idx_r in range(len(r)):
                canvas.append(r[idx_r:idx_r+1])

            batch_start = batch_end
        return canvas

    def chunker_list(seq, n):
        return [seq[i * n:(i + 1) * n] for i in range((len(seq) + n - 1) // n)]

    paths = chunker_list(paths, 8 * 3)

    for i, chunk in enumerate(paths):
        canvas = make(chunk)
        canvas = torch.cat(canvas, dim=0)

        save_path = 'make_figures/output/%s/reconstructions_%d.png' % (cfg.NAME + '_opt_randnois2_1', i)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(canvas * 0.5 + 0.5, save_path,
                   nrow=3,
                   pad_value=1.0)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-figure-reconstructions-paged', default_config='configs/ffhq.yaml',
        world_size=gpu_count, write_log=False)
