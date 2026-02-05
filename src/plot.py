import matplotlib.pyplot as plt
import torch, sys, os

def plot(title, plot_data, save_path):
	with plt.style.context("seaborn-v0_8-dark"):
		for param in ["figure.facecolor", "axes.facecolor", "savefig.facecolor"]:
			plt.rcParams[param] = "#030407"

		for param in ["text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
			plt.rcParams[param] = "0.9"

		plt.figure(figsize=(18, 8))

		for losses, label in plot_data:
			plt.plot(losses, label=label)

		plt.xlabel("steps", fontsize=12)
		plt.ylabel(label, fontsize=12)
		plt.legend(fontsize=12)
		plt.title(title, fontsize=14)
		plt.savefig(save_path, bbox_inches="tight")
		plt.close()

if not os.path.isdir("img"):
	os.mkdir("img")

checkpoint = torch.load(sys.argv[1])
stats = checkpoint["stats"]

plot("train-val loss", [(stats["loss"]["train"], "train loss"), (stats["loss"]["val"], "val loss")], f"img/train-val.png")
plot("test loss", [(stats["loss"]["test"], "test loss")], f"img/test.png")
plot("lr", [(stats["lr"], "lr")], f"img/lr.png")
