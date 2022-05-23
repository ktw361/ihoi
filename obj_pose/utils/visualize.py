import torch
import matplotlib.pyplot as plt

from obj_pose.utils.nmr_renderer import PerspectiveRenderer
from obj_pose.utils.geometry import rot6d_to_matrix



def visualize_optimal_poses(model,
                            image_crop,
                            mask,
                            score=0,
                            save_path="tmpbestobj.png"):
    """
    Visualizes the 8 best-scoring object poses.

    Args:
        model (PoseOptimizer).
        image_crop (H x H x 3).
        mask (M x M x 3).
        score (float): Mask confidence score (optional).
    """
    num_vis = 8
    rotations = model.rotations
    translations = model.translations
    verts = model.vertices[0]
    faces = model.faces[0]
    loss_dict, _, _ = model()
    losses = sum(loss_dict.values())
    camintr_roi = model.renderer.K
    inds = torch.argsort(losses)[:num_vis]
    obj_renderer = PerspectiveRenderer()

    fig = plt.figure(figsize=((10, 4)))
    ax1 = fig.add_subplot(2, 5, 1)
    ax1.imshow(image_crop)
    ax1.axis("off")
    ax1.set_title("Cropped Image")

    ax2 = fig.add_subplot(2, 5, 2)
    ax2.imshow(mask)
    ax2.axis("off")
    if score > 0:
        ax2.set_title(f"Mask Conf: {score:.2f}")
    else:
        ax2.set_title("Mask")

    for i, ind in enumerate(inds.cpu().numpy()):
        plt_ax = fig.add_subplot(2, 5, i + 3)
        rend = obj_renderer(
            vertices=verts,
            faces=faces,
            image=image_crop,
            translation=translations[ind],
            rotation=rot6d_to_matrix(rotations)[ind],
            color_name="red",
            K=camintr_roi,
        )
        plt_ax.imshow(rend)
        plt_ax.set_title(f"Rank {i}: {losses[ind]:.1f}")
        plt_ax.axis("off")
    plt.savefig(save_path)