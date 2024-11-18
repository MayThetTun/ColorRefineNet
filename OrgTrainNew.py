from torch.nn import Module
import torchvision
from torchvision import transforms
import argparse
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange
from dataloader import UWNetDataSet
from metrics_calculation import *
from newmodel import *
from combined_loss import *

__all__ = [
    "Trainer",
    "setup",
    "training",
]


@dataclass
class Trainer:
    model: Module
    opt: torch.optim.Optimizer
    loss: Module

    @torch.enable_grad()
    def train(self, train_dataloader, config, test_dataloader=None):
        device = config['device']
        primary_loss_lst = []
        vgg_loss_lst = []
        total_loss_lst = []

        # Evaluate the model on the test dataset
        UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)

        # Iterate through epochs
        for epoch in trange(0, config.num_epochs, desc=f"[Full Loop]", leave=False):
            primary_loss_tmp = 0
            vgg_loss_tmp = 0
            total_loss_tmp = 0

            # Adjust learning rate every 'step_size' epochs
            if epoch > 1 and epoch % config.step_size == 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] *= 0.7

            # Iterate through the training dataloader
            for inp, label, _ in tqdm(train_dataloader, desc=f"[Train]", leave=False):
                inp = inp.to(device)
                label = label.to(device)

                # Set the model to training mode
                self.model.train()
                self.opt.zero_grad()

                # Forward pass
                out = self.model(inp)

                # Compute loss
                loss, mse_loss, vgg_loss = self.loss(out, label)

                # Backpropagation
                loss.backward()
                self.opt.step()

                # Update temporary loss variables
                primary_loss_tmp += mse_loss.item()
                vgg_loss_tmp += vgg_loss.item()
                total_loss_tmp += loss.item()

            # Calculate average losses for the epoch
            total_loss_lst.append(total_loss_tmp / len(train_dataloader))
            vgg_loss_lst.append(vgg_loss_tmp / len(train_dataloader))
            primary_loss_lst.append(primary_loss_tmp / len(train_dataloader))

            # Print progress every 'print_freq' epochs
            if epoch % config.print_freq == 0:
                print(f'Epoch [{epoch}/{config.num_epochs}], '
                      f'Total Loss: {total_loss_lst[epoch]}, '
                      f'MSE Loss: {primary_loss_lst[epoch]}, '
                      f'VGG Loss: {vgg_loss_lst[epoch]}')

            # Evaluate on test dataset if specified and at 'eval_steps' epochs
            if config.test and epoch % config.eval_steps == 0:
                UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)

            # Save model snapshots every 'snapshot_freq' epochs
            if epoch % config.snapshot_freq == 0:
                snapshot_path = f"{config.snapshots_folder}model_epoch_{epoch}.ckpt"
                torch.save(self.model, snapshot_path)

    @torch.no_grad()
    def eval(self, config, test_dataloader, test_model):
        test_model.eval()
        for i, (img, _, name) in enumerate(test_dataloader):
            img = img.to(config["device"])
            generate_img = test_model(img)
            torchvision.utils.save_image(generate_img, config.output_images_path + name[0])

        # Calculate metrics
        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path,
                                                                   config.GTr_test_images_path)
        UIQM_measures = calculate_UIQM(config.output_images_path)
        return UIQM_measures, SSIM_measures, PSNR_measures


def setup(config):
    if torch.cuda.is_available():
        config['device'] = "cuda"
    else:

        config['device']= "cpu"

    model = UWnet(num_layers=config["num_layers"]).to(config["device"])
    transform = transforms.Compose([transforms.Resize((config["resize"], config["resize"])), transforms.ToTensor()])
    train_dataset = UWNetDataSet(config["input_images_path"], config["label_images_path"], transform, True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=False)
    print("Train Dataset Reading Completed.")

    loss = combinedloss(config)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    trainer = Trainer(model, opt, loss)

    if config["test"]:
        test_dataset = UWNetDataSet(config["test_images_path"], None, transform, False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)
        print("Test Dataset Reading Completed.")
        return train_dataloader, test_dataloader, model, trainer

    return train_dataloader, None, model, trainer


def training(config):
    config = vars(config)  # Convert argparse.Namespace to dict
    ds_train, ds_test, model, trainer = setup(config)
    trainer.train(ds_train, config, ds_test)
    print("==================")
    print("Training complete!")
    print("==================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_images_path', type=str, default='/content/drive/MyDrive/data/EUVP/train/input/',
                        help='path of input images (underwater images) default: ./data/input/')
    parser.add_argument('--label_images_path', type=str, default='/content/drive/MyDrive/data/EUVP/train/label/',
                        help='path of label images (clear images) default: ./data/label/')
    parser.add_argument('--test_images_path', type=str, default='/content/drive/MyDrive/data/EUVP/Val/input/',
                        help='path of input images (underwater images) for testing default: ./data/input/')
    parser.add_argument('--GTr_test_images_path', type=str, default='/content/drive/MyDrive/data/EUVP/Val/label/',
                        help='path of input ground truth images (underwater images) for testing default: ./data/input/')
    parser.add_argument('--test', default=True)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--step_size', type=int, default=400, help="Period of learning rate decay")
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--snapshot_freq', type=int, default=2)
    parser.add_argument('--snapshots_folder', type=str, default="./SnapShot/")
    parser.add_argument('--output_images_path', type=str, default="./ValImg/")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)

    config = parser.parse_args(args=[])
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)
    training(config)