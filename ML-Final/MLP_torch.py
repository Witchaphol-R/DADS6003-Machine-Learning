import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import argparse
import random
from Dataset import Dataset

class MLP(nn.Module):
  def __init__(self, num_users, num_items, layers, reg_layers):
    super(MLP, self).__init__()
    self.num_layers = len(layers)
    self.user_embeddings = nn.Embedding(num_users, layers[0] // 2)
    self.item_embeddings = nn.Embedding(num_items, layers[0] // 2)
    self.reg_layers = reg_layers

    mlp_layers = []
    for i in range(1, self.num_layers):
      mlp_layers.append(nn.Linear(layers[i - 1], layers[i]))
      mlp_layers.append(nn.ReLU())
    self.mlp_layers = nn.Sequential(*mlp_layers)
    self.final_layer = nn.Linear(layers[-1], 1)
    
  def forward(self, user_ids, item_ids):
    user_embeddings = self.user_embeddings(user_ids)
    item_embeddings = self.item_embeddings(item_ids)

    if user_embeddings.dim() == 2 and item_embeddings.dim() == 2:
      vector = torch.cat([user_embeddings, item_embeddings], dim=-1)
    else:
      user_embeddings = user_embeddings.expand(item_ids.size(0), -1)
      vector = torch.cat([user_embeddings, item_embeddings], dim=-1)
    
    vector = self.mlp_layers(vector)
    prediction = torch.sigmoid(self.final_layer(vector))
    return prediction

class MLPTorchDataset(Dataset):
  def __init__(self, train_matrix, num_negatives):
    self.train_matrix = train_matrix
    self.num_negatives = num_negatives
    self.users = list(set(train_matrix.nonzero()[0]))
    self.num_users, self.num_items = self.train_matrix.shape

  def __len__(self):
    return len(self.users) * (self.num_negatives + 1)

  def __getitem__(self, index):
    user = self.users[index // (self.num_negatives + 1)]
    label = 1.0 if index % (self.num_negatives + 1) == 0 else 0.0

    if label == 0:
      neg_item = np.random.randint(self.num_items)
      while (user, neg_item) in self.train_matrix:
        neg_item = np.random.randint(self.num_items)
    else:
      positive_items = self.train_matrix[user].nonzero()[1] 
      neg_item = np.random.choice(positive_items) 

    return user, neg_item, label

def evaluate(model, test_data, top_k):
  model.eval()
  hits, ndcgs = [], []
  with torch.no_grad():
    for user, item_true in test_data:
      user_input = torch.LongTensor([user]).to(device)
      item_inputs = torch.LongTensor(list(range(model.item_embeddings.num_embeddings))).to(device)
      user_input = user_input.expand(item_inputs.size(0), -1).squeeze(1)  # Repeat user_input to match item_inputs
      predictions = model(user_input, item_inputs).squeeze().cpu().numpy()
      top_k_items = predictions.argsort()[-top_k:][::-1]

      hr = 1.0 if item_true in top_k_items else 0.0
      ndcg = 0.0
      if item_true in top_k_items:
        rank = top_k_items.tolist().index(item_true) + 1
        ndcg = 1.0 / np.log2(rank + 1)

      hits.append(hr)
      ndcgs.append(ndcg)

  return np.mean(hits), np.mean(ndcgs)

def set_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', nargs='?', default='ml-1m', help='Choose a dataset.')
  parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
  parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
  parser.add_argument('--layers', nargs='?', default='[64,32,16,8]', help='Size of each layer.')
  parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]', help='Regularization for each layer.')
  parser.add_argument('--num_neg', type=int, default=4, help='Number of negative instances to pair with a positive instance.')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
  parser.add_argument('--learner', nargs='?', default='adam', help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
  parser.add_argument('--verbose', type=int, default=1, help='Show performance per X iterations')
  parser.add_argument('--out', type=int, default=1, help='Whether to save the trained model.')
  parser.add_argument('--top_k', type=int, default=10, help='Number of top recommendations to consider.')
  parser.add_argument('--seed', type=int, default=42, help='Random seed.')
  args = parser.parse_args()
  set_seed(args.seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Device:', device)

  # Load data
  data_path = 'Data/' + args.dataset 
  dataset = Dataset(data_path)
  train_dataset = MLPTorchDataset(dataset.trainMatrix, args.num_neg)
  num_users = train_dataset.num_users
  num_items = train_dataset.num_items
  layers = eval(args.layers)
  reg_layers = eval(args.reg_layers)
  model = MLP(num_users, num_items, layers, reg_layers).to(device)

  # Select optimizer
  if args.learner.lower() == "adagrad": 
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
  elif args.learner.lower() == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
  elif args.learner.lower() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
  else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

  train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

  # Training
  best_hr = 0
  for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    for user_ids, item_ids, labels in train_dataloader:
      user_ids = user_ids.to(device)
      item_ids = item_ids.to(device)
      labels = labels.to(device).float() # float32

      optimizer.zero_grad()
      predictions = model(user_ids, item_ids)
      loss = F.binary_cross_entropy(predictions, labels.view(-1, 1)) 
      loss += reg_layers[0] * torch.norm(model.user_embeddings.weight) + reg_layers[1] * torch.norm(model.item_embeddings.weight) # Add reg loss
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    if args.verbose > 0 and epoch % args.verbose == 0:
      print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_dataloader)}")

    # Evaluate on test data
    hr, ndcg = evaluate(model, [(user, item) for user, item in dataset.testRatings], args.top_k)
    print(f"Epoch {epoch+1}: HR = {hr:.4f}, NDCG = {ndcg:.4f}")

    # Save best model
    if hr > best_hr:
      best_hr = hr
      if args.out > 0: 
        torch.save(model.state_dict(), 'BestModel/best_mlp_model.pth')

  if args.out > 0:
    best_model = MLP(num_users, num_items, layers, reg_layers).to(device)
    best_model.load_state_dict(torch.load('BestModel/best_mlp_model.pth'))
    best_hr, best_ndcg = evaluate(best_model, [(user, item) for user, item in dataset.testRatings], args.top_k)
    print(f"Best HR: {best_hr:.4f}, Best NDCG: {best_ndcg:.4f}")