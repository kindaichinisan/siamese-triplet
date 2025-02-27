import torch.nn as nn
import torch.nn.functional as F
import torch

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

import torchvision.models as models

class EmbeddingNet_Resnet(nn.Module):
    def __init__(self, pretrained=True):
        super(EmbeddingNet_Resnet, self).__init__()
        
        # Load pretrained ResNet-18
        resnet_model = models.resnet18(pretrained=True)
        
        # Modify the first conv layer to accept grayscale (1-channel) images instead of RGB (3-channel)
        resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # # Freeze the layers of the ResNet model
        # for param in resnet_model.base_model.parameters():
        #     param.requires_grad = False
            
        # # Unfreeze the fully connected layer
        # for param in resnet_model.base_model.fc.parameters():
        #     param.requires_grad = True
        
        # Remove the original fully connected layers to use as a feature extractor
        self.convnet = nn.Sequential(*list(resnet_model.children())[:-1])  # Remove the final FC layer
        
        # Define new fully connected layers for embedding
        self.fc = nn.Sequential(
            nn.Linear(resnet_model.fc.in_features, 256),  # Using the ResNet feature size (usually 512 or 2048)
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 2)  # Output embedding dimension of 2
        )

    def forward(self, x):
        output = self.convnet(x)  # Pass through ResNet feature extractor
        output = output.view(output.size(0), -1)  # Flatten the features
        output = self.fc(output)  # Pass through FC layers
        return output

    def get_embedding(self, x):
        return self.forward(x)
    

# Load pre-trained ResNet18 as embedding network
class PretrainedEmbeddingNet(nn.Module):
    def __init__(self, local_weights_path=None):
        super(PretrainedEmbeddingNet, self).__init__()
        if local_weights_path is None:
            resnet = models.resnet18(pretrained=True)  # Load pre-trained model
        else:
            resnet = models.resnet18(pretrained=False)  # Don't download from internet
            resnet.load_state_dict(torch.load(local_weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # Flatten output