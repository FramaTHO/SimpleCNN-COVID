
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import cv2
import argparse

from gradcam import GradCam 
from data.data_manager import DataManager
from loss.sord_function import sord_function
from eval import evaluate, _compute_scores

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # last convolution
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(18496, 1000) 
        self.fc2 = nn.Linear(1000, 4)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def process_explanations_train(expl, n_batches):
    tmp_list = []
    for _ in range(n_batches):
        image = cv2.resize(expl, (14,14), cv2.INTER_AREA)
        tmp_list.append(torch.from_numpy(image/255))
    return torch.stack(tmp_list).double()

def process_explanations_eval(expl, n_batches):
    tmp_list = []
    for _ in range(n_batches):
        tmp_list.append(expl/255)
    return tmp_list

def calculate_measures(gts, masks):
    final_prec = 0
    final_rec = 0

    for mask, gt in zip(masks, gts): 
        if (gt.sum() == 0) | (mask.sum() == 0): ## DA SOSTITUIRE CON QUALCOSA DI MIGLIORE
            precision = 0
            recall = 0
        else:
            precision = np.sum(gt*mask) / (np.sum(gt*mask) + np.sum((1-gt)*mask))
            recall = np.sum(gt*mask) / (np.sum(gt*mask) + np.sum(gt*(1-mask)))

        final_prec = final_prec + precision
        final_rec = final_rec + recall 

    return final_prec, final_rec


def training_model(model, train_loader, bce, cross_entropy, optimizer, epoch, num_epochs, n_batches, writer):
    lamb = 1
    all_labels = []
    all_prediction = []
    model.train()
    for i, (images, metadata) in enumerate(train_loader):
            masks = grad_cam(images, training=True)
            model.train()

            ### WRITE FIRST 6 EXPLANATIONS. --- DEBUGGING PURPUOSE --- ###
            #if i == 0:
            #    os.mkdir('./explanations/epoch_'+str(epoch))
            #    temp_npy = [masks[j].cpu().detach().numpy() for j in range(6)] # lista di numpy arrays
            #    for idx, mask in enumerate(temp_npy):
            #        mask = np.float32(mask*255)
            #        mask = np.uint8(np.around(mask,decimals=0))
            #        th, dst = cv2.threshold(mask, 200, 225, cv2.THRESH_BINARY)
            #        cv2.imwrite('./explanations/epoch_'+str(epoch)+'/expl_'+str(idx)+'_ep_'+str(epoch)+'_.png', dst)
            ### WRITE FIRST 6 EXPLANATIONS. --- DEBUGGING PURPUOSE --- ###

            tensors_images = []
            for hospital, expl in zip(metadata["hospital"], metadata["explanation"]):
                np_image = np.load("/home/dataset/segmentation frames/"+hospital+"/"+expl)
                resized_image = cv2.resize(np_image, (35,35), cv2.INTER_AREA) ########################
                norm_image = np.where(resized_image > 0, 1, 0)
                norm_image = resized_image / 255
                tensors_images.append(torch.from_numpy(norm_image)) 
            
            groundT = torch.stack(tensors_images).double()    

            optimizer.zero_grad()
            loss_gradcam = bce(masks, groundT)
            loss_gradcam = loss_gradcam.cuda()
            loss_label, label_multiclass, class_predictions = sord_function(model, images, metadata)
            loss = (lamb * loss_label) + ((1 - lamb) * loss_gradcam)
            loss.backward()
            optimizer.step()

            all_labels.append(label_multiclass)
            all_prediction.append(class_predictions)

            nump_masks = masks.cpu().detach().numpy()
            list_of_expl_npy = groundT.cpu().detach().numpy()
            prec, rec = calculate_measures(list_of_expl_npy, nump_masks)

            prec = prec / n_batches
            rec = rec / n_batches
  
            if (i + 1) % 660 == 0:    
                all_labels = np.concatenate(all_labels)
                all_predictions = np.concatenate(all_predictions) 
                scores = _compute_scores(all_labels, all_predictions)           
                print('Epoch [{}/{}], Step [{}/{}], Total Loss: {:.4f}, LABELS:[F1: {:.2f}%], EXPLS:[precision: {:.2f}%, recall: {:.2f}%]'
                     .format((epoch + 1), num_epochs, (i + 1), len(train_loader), loss.item(), scores["test/f1"]*100, prec*100, rec*100))

                writer.add_scalar("Training: Loss", loss.item(), str(epoch + 1)+'_'+str(i+1))
                writer.add_scalar("Training: Precision", prec*100, str(epoch + 1)+'_'+str(i+1))
                writer.add_scalar("Training: Recall", rec*100, str(epoch + 1)+'_'+str(i+1))
                writer.add_scalar("Training: F1", scores["test/f1"]*100, str(epoch + 1)+'_'+str(i+1))
    if epoch % 10 == 0:           
        torch.save(model.state_dict(), "./checkpoints/simpleCNN_"+str(epoch+1)+".pth")  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument('--hospitals', nargs='+', type=str, default=['Germania', 'Pavia', 'Lucca', 'Brescia', 'Gemelli - Roma', 'Tione', 'Trento'], 
        help='Name of the hospital / folder to be used.')
    parser.add_argument('--dataset_root', default='/home/dataset/', type=str, help='Root folder for the datasets.')
    parser.add_argument('--split_file', default='80_20_activeset.csv', type=str, help='File defining train and test splits.')
    parser.add_argument('--standard_image_size', nargs='+', type=int, default=[250, 250])
    parser.add_argument('--input_image_size', nargs='+', type=int, default=[70, 70]) 
    parser.add_argument('--domains_count', type=int, default=2)
    parser.add_argument('--domain_label', type=str, default="sensor_label")
    parser.add_argument('--affine_sigma', type=float, default=0.0)
    parser.add_argument('--rotation', type=float, default=23.0)
    # Environment
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--num_workers", default=os.cpu_count() // 2, type=int)
    parser.add_argument('--test_size', default=0.3, type=float, help='Relative size of the test set.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--split', default='patient_hash', type=str, help='The split strategy.')
    parser.add_argument('--stratify', default=None, type=str, help='The field to stratify by.')
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=33, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=3, help="interval evaluations on validation set")
    # Network
    parser.add_argument("--batch_size", default=64, type=int)
    opt = parser.parse_args()

    writer = SummaryWriter("./runs/")
    
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001

    cuda = torch.cuda.is_available()
    if cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU...')

    model = Net() 
    # Start from checkpoint, if specified
    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights))
        print("pretrained model loaded!")    
    if cuda:
        model = model.cuda()
        print('Loaded model on GPU')

    data_manager = DataManager(opt) 
    dataset = data_manager.get_datasets()
    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]

    train_dataloader = data_manager.get_dataloaders()["train"]
    test_dataloader = data_manager.get_dataloaders()["validation"]

    grad_cam= GradCam(model=model, feature_module=model.layer2, \
                      target_layer_names=["0"], use_cuda=True)

    bce = nn.BCELoss()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        training_model(model, train_dataloader, bce, cross_entropy, optimizer, epoch, num_epochs, batch_size, writer)
        #if (epoch+1) % 10 == 0:
            #evaluate(model, test_dataloader, writer, epoch)