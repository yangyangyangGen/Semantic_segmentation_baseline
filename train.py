from modules.pspnet import PSPNet
from modules.fcn8s import FCN8s
from modules.deeplab import DeepLabV3
from modules.unet import UNet
from modules.transforms import *
from modules.loss import CrossEntropy
from modules.dataloader import BasicDataLoader
from modules.utils import AverageMeter
from modules.metrics_matrix import SegmentationMetrics
from functools import partial

import os
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='PSPNet')
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--step_interval', type=int, default=10)
    parser.add_argument('--ignore_label', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=2)
    parser.add_argument('--train_image_folder', type=str,
                        default=r'/home/aistudio/data/data56259')
    parser.add_argument('--train_image_list_file', type=str,
                        default='/home/aistudio/data/data56259/CelebAMask-HQ/train_list.txt')
    parser.add_argument('--valid_image_folder', type=str,
                        default=r'/home/aistudio/data/data56259')
    parser.add_argument('--valid_image_list_file', type=str,
                        default='/home/aistudio/data/data56259/CelebAMask-HQ/val_list.txt')
    parser.add_argument('--checkpoint_folder', type=str, default='./output')
    parser.add_argument('--save_freq', type=int, default=1)

    return parser.parse_args()


global_model_dict = {
    "PSPNet": PSPNet,
    "FCN8s": FCN8s,
    "DeepLabV3": DeepLabV3,
    "UNet": UNet
}

args = parse_args()
seg_metrix = SegmentationMetrics(args.num_classes, args.ignore_label)


def train(dataloader, model, criterion, optimizer, epoch, total_batch, ncls, step_interval):
    model.train()
    train_loss_meter = AverageMeter()
    for batch_id, (data, target) in enumerate(dataloader):
        data = fluid.dygraph.to_variable(data)
        target = fluid.dygraph.to_variable(target)
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.minimize(loss)
        optimizer.clear_gradients()

        n = data.shape[0]
        train_loss_meter.update(loss.numpy()[0], n)

        if (batch_id + 1) % step_interval == 0:
            print("Train: " +
                  f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
                  f"Step[{batch_id:04d}/{total_batch:04d}], " +
                  f"Average Loss: {train_loss_meter.avg:4f}")

    return train_loss_meter.avg


def valid(dataloader, model, criterion, epoch, total_batch, ncls):
    model.eval()
    valid_loss_meter = AverageMeter()
    valid_acc_meter = AverageMeter()
    valid_miou_meter = AverageMeter()
    valid_mpa_meter = AverageMeter()

    for batch_id, (data, target) in enumerate(dataloader):
        data = fluid.dygraph.to_variable(data)
        target = fluid.dygraph.to_variable(target)

        pred = model(data)
        loss = criterion(pred, target)

        info = seg_metrix(pred.numpy(), target.numpy().squeeze())
        valid_acc_meter.update(info["acc"], 1)
        valid_miou_meter.update(
            float(sum(info["miou"])) / len(info["miou"]), 1)
        valid_mpa_meter.update(
            float(sum(info["mpa"])) / len(info["mpa"]), 1)
        n = data.shape[0]
        valid_loss_meter.update(loss.numpy()[0], n)

    print("Valid: " +
          f"Epoch[{epoch:03d}/{args.num_epochs:03d}], \n" +
          f"Step[{batch_id:04d}/{total_batch:04d}], \n" +
          f"Average Acc: {valid_acc_meter.avg:4f}, \n" +
          f"Average Loss: {valid_loss_meter.avg:4f}, \n" +
          f"Average Miou: {valid_miou_meter.avg:4f}, \n" +
          f"Average Mpa:  {valid_mpa_meter.avg:4f}")

    return (valid_loss_meter.avg, valid_acc_meter.avg,
            valid_miou_meter.avg, valid_mpa_meter.avg)


def main():
    global global_model_dict
    global args
    print(f"Using net is {args.net}")
    # Step 0: preparation
    with fluid.dygraph.guard():
        # Step 1: Define training dataloader
        size = 512
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transforms = Compose(
            # RandomCrop(size),
            Resize(size),
            RandomHorizontalFlip(),
            Normalize(mean, std),
            ConvertDataType(),
            HWC2CHW()
        )
        valid_transforms = Compose(
            Resize(size),
            Normalize(mean, std),
            ConvertDataType(),
            HWC2CHW()
        )

        train_dataset = BasicDataLoader(args.train_image_folder,
                                        args.train_image_list_file,
                                        train_transforms,
                                        shuffle=True)

        valid_dataset = BasicDataLoader(args.valid_image_folder,
                                        args.valid_image_list_file,
                                        valid_transforms,
                                        shuffle=False)

        train_total_batch = len(train_dataset) // args.train_batch_size
        valid_total_batch = len(valid_dataset) // args.valid_batch_size

        # define dataloader.
        train_dataloader = fluid.io.DataLoader.from_generator(
            capacity=1, use_multiprocess=False)
        valid_dataloader = fluid.io.DataLoader.from_generator(
            capacity=1, use_multiprocess=False)

        train_dataloader.set_sample_generator(train_dataset,
                                              args.train_batch_size,
                                              drop_last=False)
        valid_dataloader.set_sample_generator(valid_dataset,
                                              args.valid_batch_size,
                                              drop_last=False)

        assert args.net in global_model_dict.keys(), \
            f"{args.net} not in {global_model_dict.keys()}"

        # Step 2: Create model
        model = global_model_dict[args.net](3, args.num_classes)

        # Step 3: Define criterion and optimizer
        # criterion = partial(CrossEntropy, ignore_index=args.ignore_label)
        criterion = CrossEntropy

        # create optimizer
        optimizer = AdamOptimizer(args.lr,
                                  parameter_list=model.parameters())

        # Step 4: Training
        best_valid_loss = 0.
        for i, epoch in enumerate(range(1, args.num_epochs+1)):
            train_loss = train(train_dataloader,
                               model,
                               criterion,
                               optimizer,
                               epoch,
                               train_total_batch,
                               args.num_classes,
                               args.step_interval)

            valid_metric = valid(valid_dataloader,
                                 model,
                                 criterion,
                                 epoch,
                                 valid_total_batch,
                                 args.num_classes)

            valid_loss, valid_acc, valid_miou, valid_mpa = valid_metric

            if i == 0:
                best_valid_loss = valid_loss

            print(
                f"""----- Epoch[{epoch}/{args.num_epochs}]
                Train Loss: {train_loss: .4f}, 
                Valid Loss: {valid_loss: .4f}""")

            if (epoch % args.save_freq == 0 or epoch == args.num_epochs) and valid_loss < best_valid_loss:

                model_path = os.path.join(
                    args.checkpoint_folder, args.net, f"Epoch-{epoch}-Loss-{valid_loss}")

                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                # save model and optmizer states
                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)
                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')

                best_valid_loss = valid_loss


if __name__ == "__main__":
    main()
