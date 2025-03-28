import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from time import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def autoencoder_train_loop(model, dataloader, criterion,
                           optimizer, device, epochs):
    loss_list = []
    model.train()
    with tqdm(range(epochs)) as pbar:
        for _ in pbar:
            loss_train = 0
            for y_true, _ in dataloader:
                y_true = y_true.to(device)
                y_pred = model(y_true)
                loss = criterion(y_pred, y_true)
                loss_train += (loss_value := loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss_value)
            loss_list.append(loss_train / len(dataloader))
    return loss_list


def plot_loss(epochs, loss_dict, config_dict, ax=None, colors=None, title=None):
    if loss_dict:
        if ax is None:
            ax = plt.gca()
        if colors is None:
            colors = [None] * (len(loss_dict))
        x_axis = range(1, epochs + 1)
        for (loss_type, loss), color in zip(loss_dict.items(), colors):
            ax.plot(x_axis, loss, label=f'{loss_type}: {min(loss):.3f}', color=color)
        ax.set(title=title, xlabel='Epoch', ylabel='Loss', **config_dict)
        ax.legend()


def sample_random_digits(mnist_dataset, seed=42):
    torch.manual_seed(seed)
    targets = mnist_dataset.targets.numpy()
    samples = {}
    for digit in range(10):
        indices = [i for i, label in enumerate(targets) if label == digit]
        chosen_idx = np.random.choice(indices)
        samples[digit] = mnist_dataset[chosen_idx][0]
    return samples


def compare_generated_to_original_mnist(model_dict, mnist_dataset, cmap='gray'):
    n_rows = len(model_dict) + 1
    figure, axes = plt.subplots(n_rows, 10, figsize=(15, 2 * n_rows))
    image_dict = sample_random_digits(mnist_dataset, seed=42)
    for digit, img in image_dict.items():
        axes[0, digit].imshow(img.permute(1, 2, 0).numpy(), cmap=cmap)
        for i, model in enumerate(model_dict.values(), start=1):
            model.eval()
            with torch.inference_mode():
                reconstruction = model.cpu()(img.unsqueeze(0))
                if isinstance(reconstruction, tuple):
                    reconstruction = reconstruction[0]
                reconstruction = reconstruction.squeeze(0)
                axes[i, digit].imshow(reconstruction.permute(1, 2, 0).numpy(), cmap=cmap)
    for ax in axes.ravel():
        ax.axis(False)
    for i, model_name in enumerate(['Original'] + list(model_dict.keys())):
        pos = axes[i, 0].get_position()
        x_coord = pos.y1 + (pos.y1 - pos.y0) / 5
        figure.text(0.5, x_coord, f'Architecture: {model_name}',
                    ha='center', va='center', fontsize=18)
    return figure, axes


def get_latent_space_points(encoder, dataset, batch_size=256, device='cuda'):
    encoder = encoder.to(device).eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    points_list = []
    label_list = []
    with torch.inference_mode():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            points = encoder(images).cpu().numpy()
            points_list.append(points)
            label_list.append(labels.numpy())
    return np.vstack(points_list).T, np.hstack(label_list)


def get_dict_of_latent_spaces(autoencoder_dict, dataset, batch_size=256, device='cuda'):
    latent_space_dict = {}
    for name, model in autoencoder_dict.items():
        latent_space_dict[name] = get_latent_space_points(
            model.encoder, dataset, batch_size=batch_size, device=device)
    return latent_space_dict


def compare_latent_spaces(latent_space_dict, axes_list,
                          s=0.5, alpha=0.7, cmap='cool'):
    for (name, space), ax in zip(latent_space_dict.items(), axes_list.ravel()):
        latent_points, labels = space
        ax.scatter(*latent_points, c=labels, s=s, alpha=alpha, cmap=cmap)
        ax.axis(False)
        ax.set(title=name)


def zoom_limits(data, zoom_factor):
    data_min, data_max = np.min(data), np.max(data)
    range_center = (data_max + data_min) / 2
    range_half = (data_max - data_min) * zoom_factor / 2
    return range_center - range_half, range_center + range_half


def compare_3d_latent_spaces(latent_space_dict, axes_list,
                             s=0.5, alpha=0.7, cmap='cool', zoom=0.85):
    for i, ((name, space), ax) in enumerate(zip(latent_space_dict.items(), axes_list.ravel()),
                                                  start=1):
        latent_points, labels = space
        ax.scatter(*latent_points, c=labels, s=s, alpha=alpha, cmap=cmap)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_tick_params(which='major', color='black')
        ax.set(title=name)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlim(zoom_limits(latent_points[0], zoom_factor=zoom))
        ax.set_ylim(zoom_limits(latent_points[1], zoom_factor=zoom))
        ax.set_zlim(zoom_limits(latent_points[2], zoom_factor=zoom))


def dataset_subset(dataset, n_samples):
    return Subset(dataset, torch.randperm(len(dataset))[:n_samples])


def reduce_dimensions(latent_space, reduced_dim=2, method='PCA', **method_params):
    start = time()
    match method.upper():
        case 'PCA':
            reduction = PCA(n_components=reduced_dim, **method_params)
        case 'TSNE':
            reduction = TSNE(n_components=reduced_dim, **method_params)
        # imcompatible with python 3.12
        # case 'TSNE_CUDA':
        #     reduction = TSNE_cuda(n_components=reduced_dim, **method_params)
        case 'UMAP':
            reduction = UMAP(n_components=reduced_dim, **method_params)
        case _:
            return None
    result = reduction.fit_transform(latent_space)
    total_time = time() - start
    return result, total_time


def reduce_dimensions_of_dict(latent_space_dict,
                              reduced_dim=2, method='PCA', **method_params):
    reduced_dim_dict = {}
    for name, (latent_space, latent_space_label) in latent_space_dict.items():
        result, result_time = reduce_dimensions(
            latent_space.T, reduced_dim=reduced_dim, method=method, **method_params)
        reduced_dim_dict[name] = (result.T, latent_space_label)
        print(f'Reduced for {name}, time taken: {result_time:.3f} seconds.')
    return reduced_dim_dict


def vae_train_loop(model, dataloader, criterion,
                   optimizer, device, epochs):
    recon_loss_list = []
    kl_loss_list = []
    total_loss_list = []
    model.train()
    with tqdm(range(epochs)) as pbar:
        for _ in pbar:
            recon_loss_train = 0
            kl_loss_train = 0
            total_loss_train = 0
            for x_true, _ in dataloader:
                x_true = x_true.to(device)
                x_pred, mu, logvar = model(x_true)
                total_loss, recon_loss, kl_loss = criterion(x_pred, x_true, mu, logvar)
                optimizer.zero_grad()
                total_loss.backward()
                recon_loss_train += (recon_loss_value := recon_loss.item())
                kl_loss_train += (kl_loss_value := kl_loss.item())
                total_loss_train += (total_loss_value := total_loss.item())
                optimizer.step()
                pbar.set_postfix(dict(loss=f'{total_loss_value:.3f}',
                                      rec=f'{recon_loss_value:.3f}',
                                      kl=f'{kl_loss_value:.5f}'))
            total_loss_list.append(total_loss_train / (data_len := len(dataloader)))
            recon_loss_list.append(recon_loss_train / data_len)
            kl_loss_list.append(kl_loss_train / data_len)
    return total_loss_list, recon_loss_list, kl_loss_list
