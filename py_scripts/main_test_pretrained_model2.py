import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import cv2
import numpy as np
import os

# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet, ClassificationNet, PretrainedEmbeddingNet
from losses import ContrastiveLoss

import utils


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        images_list=[]
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            print(f'images.shape: {images.shape}')
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
            images_list.append(images)
    return embeddings, labels, images_list

if __name__ == '__main__':

    # mean, std = 0.28604059698879553, 0.35302424451492237
    # batch_size = 256

    # train_dataset = FashionMNIST('../data/FashionMNIST', train=True, download=True,
    #                             transform=transforms.Compose([
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((mean,), (std,))
    #                             ]))

    # cuda = torch.cuda.is_available()
    # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # n_classes = 10

    # embedding_net = EmbeddingNet()
    # # model = ClassificationNet(embedding_net, n_classes=n_classes)
    # model = SiameseNet(embedding_net)
    # if cuda:
    #     model.cuda()
    # train_embeddings_baseline, train_labels_baseline, images_list = extract_embeddings(train_loader, model)

    # print(len(images_list))
    # print(images_list[0].shape)

    # output1, output2 = model(images_list[0],images_list[0])

    # print(output1, output2)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    imgfilepath1=r'C:\Users\hwenjun\Desktop\test_img\563766000.png'
    imgfilepath2=r'C:\Users\hwenjun\Desktop\test_img\255806425.png'
    cuda = torch.cuda.is_available()
    print(cuda)

    # Step 2
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, r'..\model\resnet18-f37072fd.pth')
    embedding_net = PretrainedEmbeddingNet(weights_path) #EmbeddingNet()
    # Step 3
    model = SiameseNet(embedding_net)
    if cuda:
        model.cuda()

    # img1=cv2.imread(imgfilepath1)
    # img2=cv2.imread(imgfilepath2)

    img1=utils.preprocess_image(imgfilepath1)
    img2=utils.preprocess_image(imgfilepath2)
    # img1=cv2.resize(img1, (28, 28))
    # img2=cv2.resize(img2, (28, 28))

    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # img1=torch.from_numpy(img1).float()
    # img2=torch.from_numpy(img2).float()

    # print(img1.shape)
    # img1=img1.unsqueeze(0).unsqueeze(0)
    # img2=img2.unsqueeze(0).unsqueeze(0)
    #img1 = img1.expand(256, -1, -1, -1)
    img1=img1.cuda()
    img2=img2.cuda()
    print(img1.shape)
    embedding1, embedding2 = model(img1, img2)

    # Compute Euclidean distance
    print(embedding1)
    distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
    print("Euclidean Distance:", distance.item())

    # Compute Cosine Similarity
    cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    print("Cosine Similarity:", cosine_sim.item())

    # Apply threshold
    THRESHOLD = 0.5  # Adjust based on your dataset
    if distance.item() < THRESHOLD:
        print("Images are similar")
    else:
        print("Images are different")