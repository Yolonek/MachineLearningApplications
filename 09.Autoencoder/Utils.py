from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import numpy as np


def autoencoder_train_loop(model, dataloader, criterion,
                           optimizer, device, epochs):
    loss_list = []
    model.train()
    with tqdm(range(epochs)) as pbar:
        for _ in pbar:
            loss_train = 0
            for images, _ in dataloader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
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
            reconstruction = model.cpu()(img.unsqueeze(0)).squeeze(0).detach()
            axes[i, digit].imshow(reconstruction.permute(1, 2, 0).numpy(), cmap=cmap)
    for ax in axes.ravel():
        ax.axis(False)
    for i, model_name in enumerate(['Original'] + list(model_dict.keys())):
        pos = axes[i, 0].get_position()
        x_coord = pos.y1 + (pos.y1 - pos.y0) / 5
        figure.text(0.5, x_coord, f'Architecture: {model_name}',
                    ha='center', va='center', fontsize=18)
    return figure, axes


def get_latent_space_points(model, dataset):
    model = model.cpu().eval()
    points = []
    label_list = []
    for image, label in tqdm(dataset):
        point = model(image.unsqueeze(0)).squeeze(0).detach().numpy()
        points.append(point)
        label_list.append(label)
    return np.swapaxes(np.vstack(points), 1, 0), label_list


def compare_latent_spaces(models_dict, dataset, axes_list,
                          s=0.5, alpha=0.7, cmap='cool'):
    for (model_name, model), ax in zip(models_dict.items(), axes_list.ravel()):
        latent_points, labels = get_latent_space_points(model.encoder, dataset)
        ax.scatter(*latent_points, c=labels, s=s, alpha=alpha, cmap=cmap)
        ax.axis(False)
        ax.set(title=model_name)


def zoom_limits(data, zoom_factor):
    data_min, data_max = np.min(data), np.max(data)
    range_center = (data_max + data_min) / 2
    range_half = (data_max - data_min) * zoom_factor / 2
    return range_center - range_half, range_center + range_half

