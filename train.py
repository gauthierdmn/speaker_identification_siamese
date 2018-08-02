from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
from net import SiameseNetwork
from data_loader import SiameseDataset
from contrastive import ContrastiveLoss
import pickle

num_epochs = 10
batch_size = 30
lr = 0.001
cuda = False

with open("labels/siamese_labels.pkl", "rb") as f:
    labels = pickle.load(f)

siamese_dataset = SiameseDataset("audio", labels)

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        batch_size=batch_size)

model = SiameseNetwork()
if cuda:
    model.cuda()

criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

counter = []
loss_history = []
iteration_number = 0

model.train()
for epoch in range(0,num_epochs):
    print("##### epoch {:2d}".format(epoch))
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = Variable(img0).unsqueeze(1), Variable(img1).unsqueeze(1), Variable(label)
        if cuda:
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        output1, output2 = model(img0,img1)
        optimizer.zero_grad()
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()

        if i % 1 == 0 :
            print("Current loss: {}".format(loss_contrastive.data[0]))
            iteration_number +=1
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data[0])
