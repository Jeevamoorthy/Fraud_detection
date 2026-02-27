import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os
import random
import numpy as np

# Determinism
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(FraudGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.fc = torch.nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.fc(x)
        return torch.sigmoid(x)

class ModelHandler:
    def __init__(self, model_path="c:/Users/jeeva/Desktop/fraud_detection/MuleNet/models/trained_gnn.pt"):
        self.model_path = model_path
        # 9 features from FeatureEngineer
        self.model = FraudGNN(in_channels=9, hidden_channels=32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.BCELoss()
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print("Loaded pre-trained GNN model (eval mode).")
        else:
            # Save initialized weights so it's consistent if no model exists yet
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model.state_dict(), model_path)
            self.model.eval()
            print("Saved initial weights and set to eval mode.")

    def train(self, data, epochs=50):
        """
        data: PyG Data object.
        Expected attributes: x (node features), edge_index, y (labels), train_mask.
        """
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index).squeeze()
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
                
        torch.save(self.model.state_dict(), self.model_path)
        print("Model saved.")

    def predict(self, x, edge_index):
        """Real-time inference."""
        self.model.eval()
        with torch.no_grad():
            prob = self.model(x, edge_index).squeeze()
        return prob.item() if prob.dim() == 0 else prob.numpy()

if __name__ == "__main__":
    # Dummy verification
    handler = ModelHandler()
    x = torch.randn((10, 9))
    edge_index = torch.randint(0, 10, (2, 20))
    print("Initial prediction sample:", handler.predict(x, edge_index))
