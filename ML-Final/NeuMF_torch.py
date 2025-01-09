import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import random
from Dataset import Dataset

class GMF(nn.Module):
  def __init__(self, num_users, num_items, embedding_dim, regs=[0, 0]):
    super(GMF, self).__init__()
    self.user_embeddings = nn.Embedding(num_users, embedding_dim)
    self.item_embeddings = nn.Embedding(num_items, embedding_dim)
    self.regs = regs

  def forward(self, user_ids, item_ids):
    user_embeddings = self.user_embeddings(user_ids)
    item_embeddings = self.item_embeddings(item_ids)
    interaction = torch.sum(user_embeddings * item_embeddings, dim=1)
    return torch.sigmoid(interaction).view(-1, 1)

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


class NeuMF(nn.Module):
  def __init__(self, num_users, num_items, mf_dim, layers, reg_layers, reg_mf):
    super(NeuMF, self).__init__()
    self.num_layers = len(layers)
    
    # GMF part
    self.mf_user_embeddings = nn.Embedding(num_users, mf_dim)
    self.mf_item_embeddings = nn.Embedding(num_items, mf_dim)
    
    # MLP part
    self.mlp_user_embeddings = nn.Embedding(num_users, layers[0] // 2)
    self.mlp_item_embeddings = nn.Embedding(num_items, layers[0] // 2)
    
    mlp_layers = []
    for i in range(1, self.num_layers):
      mlp_layers.append(nn.Linear(layers[i - 1], layers[i]))
      mlp_layers.append(nn.ReLU())
    self.mlp_layers = nn.Sequential(*mlp_layers)
    
    # Final prediction layer
    self.predict_layer = nn.Linear(mf_dim + layers[-1], 1)
    
  def forward(self, user_ids, item_ids):
    # GMF part
    mf_user_embeddings = self.mf_user_embeddings(user_ids)
    mf_item_embeddings = self.mf_item_embeddings(item_ids)
    mf_vector = mf_user_embeddings * mf_item_embeddings
    
    # MLP part
    mlp_user_embeddings = self.mlp_user_embeddings(user_ids)
    mlp_item_embeddings = self.mlp_item_embeddings(item_ids)

    if mlp_user_embeddings.shape != mlp_item_embeddings.shape:
      print(f"mlp_user_embeddings: {mlp_user_embeddings.shape}")
      print(f"mlp_item_embeddings: {mlp_item_embeddings.shape}")
    
    if mlp_user_embeddings.dim() == 2 and mlp_item_embeddings.dim() == 2:
      mlp_vector = torch.cat([mlp_user_embeddings, mlp_item_embeddings], dim=-1)
    else:
      mlp_user_embeddings = mlp_user_embeddings.unsqueeze(1).expand(-1, item_ids.size(0), -1)
      mlp_user_embeddings = mlp_user_embeddings.view(-1, mlp_user_embeddings.size(-1))
      mlp_item_embeddings = mlp_item_embeddings.view(-1, mlp_item_embeddings.size(-1))
      mlp_vector = torch.cat([mlp_user_embeddings, mlp_item_embeddings], dim=-1)
    mlp_vector = self.mlp_layers(mlp_vector)
    
    # Concatenate GMF and MLP parts
    predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
    prediction = torch.sigmoid(self.predict_layer(predict_vector))
    return prediction

class NeuMFDataset(Dataset):
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
      item_inputs = torch.LongTensor(list(range(model.mf_item_embeddings.num_embeddings))).to(device)
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

def load_pretrained_model(model, gmf_model, mlp_model, num_layers):
  # GMF embeddings
  model.mf_user_embeddings.weight.data.copy_(gmf_model.user_embeddings.weight.data)
  model.mf_item_embeddings.weight.data.copy_(gmf_model.item_embeddings.weight.data)
  
  # MLP embeddings
  model.mlp_user_embeddings.weight.data.copy_(mlp_model.user_embeddings.weight.data)
  model.mlp_item_embeddings.weight.data.copy_(mlp_model.item_embeddings.weight.data)
  
  # MLP layers
  for i in range(1, num_layers):
    if isinstance(model.mlp_layers[i - 1], nn.Linear):
      model.mlp_layers[i - 1].weight.data.copy_(mlp_model.mlp_layers[i - 1].weight.data)
      model.mlp_layers[i - 1].bias.data.copy_(mlp_model.mlp_layers[i - 1].bias.data)
  
  # Prediction layer
  mlp_final_weight = mlp_model.final_layer.weight.data
  mlp_final_bias = mlp_model.final_layer.bias.data
  gmf_final_weight = torch.zeros_like(mlp_final_weight[:, :model.mf_user_embeddings.embedding_dim])
  gmf_final_bias = torch.zeros_like(mlp_final_bias)

  model.predict_layer.weight.data.copy_(torch.cat([gmf_final_weight, mlp_final_weight], dim=1))
  model.predict_layer.bias.data.copy_(gmf_final_bias + mlp_final_bias)
  
  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', nargs='?', default='ml-1m', help='Choose a dataset.')
  parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
  parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
  parser.add_argument('--num_factors', type=int, default=8, help='Embedding size of MF model.')
  parser.add_argument('--layers', nargs='?', default='[64,32,16,8]', help='MLP layers.')
  parser.add_argument('--reg_mf', type=float, default=0, help='Regularization for MF embeddings.')                    
  parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]', help='Regularization for each MLP layer.')
  parser.add_argument('--num_neg', type=int, default=4, help='Number of negative instances to pair with a positive instance.')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
  parser.add_argument('--learner', nargs='?', default='adam', help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
  parser.add_argument('--verbose', type=int, default=1, help='Show performance per X iterations')
  parser.add_argument('--mf_pretrain', nargs='?', default='', help='Specify the pretrain model file for MF part.')
  parser.add_argument('--mlp_pretrain', nargs='?', default='', help='Specify the pretrain model file for MLP part.')
  parser.add_argument('--top_k', type=int, default=10, help='Number of top recommendations to consider.')
  parser.add_argument('--out', type=int, default=1, help='Whether to save the trained model.')
  parser.add_argument('--seed', type=int, default=22004, help='Random seed.')
  args = parser.parse_args()
  set_seed(args.seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Device:', device)

  # Load data
  data_path = 'Data/' + args.dataset 
  dataset = Dataset(data_path)
  train_dataset = NeuMFDataset(dataset.trainMatrix, args.num_neg)
  num_users = train_dataset.num_users
  num_items = train_dataset.num_items
  layers = eval(args.layers)
  reg_layers = eval(args.reg_layers)
  model = NeuMF(num_users, num_items, args.num_factors, layers, reg_layers, args.reg_mf).to(device)

  # Select optimizer
  if args.learner.lower() == "adagrad": 
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
  elif args.learner.lower() == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
  elif args.learner.lower() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
  else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

  # Load pretrain model
  if args.mf_pretrain and args.mlp_pretrain:
    gmf_model = GMF(num_users, num_items, args.num_factors)
    gmf_model.load_state_dict(torch.load(args.mf_pretrain))
    mlp_model = MLP(num_users, num_items, layers, reg_layers)
    mlp_model.load_state_dict(torch.load(args.mlp_pretrain))
    model = load_pretrained_model(model, gmf_model, mlp_model, len(layers))
    print(f"Loaded pretrained GMF model from {args.mf_pretrain} and MLP model from {args.mlp_pretrain}")

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
      loss += args.reg_mf * torch.norm(model.mf_user_embeddings.weight) + args.reg_mf * torch.norm(model.mf_item_embeddings.weight)
      for reg_layer, layer in zip(reg_layers, model.mlp_layers):
        if isinstance(layer, nn.Linear):
          loss += reg_layer * torch.norm(layer.weight)
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
        torch.save(model.state_dict(), 'BestModel/best_neumf_model.pth')

  if args.out > 0:
    best_model = NeuMF(num_users, num_items, args.num_factors, layers, reg_layers, args.reg_mf).to(device)
    best_model.load_state_dict(torch.load('BestModel/best_neumf_model.pth'))
    best_hr, best_ndcg = evaluate(best_model, [(user, item) for user, item in dataset.testRatings], args.top_k)
    print(f"Best HR: {best_hr:.4f}, Best NDCG: {best_ndcg:.4f}")