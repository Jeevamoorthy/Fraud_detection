import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import HeteroData

class GraphBuilder:
    def __init__(self):
        self.nx_graph = nx.MultiDiGraph()
        
    def build_from_dataframe(self, df):
        """Constructs a NetworkX graph from a transactions DataFrame."""
        self.nx_graph.clear()
        
        for _, row in df.iterrows():
            tx_id = row['transaction_id']
            # Nodes
            from_acc = f"ACC_{row['from_account']}"
            self.nx_graph.add_node(from_acc, type='account')
            
            if pd.notna(row.get('to_account')):
                to_acc = f"ACC_{row['to_account']}"
                self.nx_graph.add_node(to_acc, type='account')
                self.nx_graph.add_edge(from_acc, to_acc, key=tx_id, type='TRANSFERRED_TO', 
                                       amount=row['amount'], timestamp=row['timestamp'], channel=row['channel'])
            
            if pd.notna(row.get('device_id')):
                device = f"DEV_{row['device_id']}"
                self.nx_graph.add_node(device, type='device')
                self.nx_graph.add_edge(from_acc, device, key=f"{tx_id}_dev", type='USED_DEVICE', timestamp=row['timestamp'])
                
            if pd.notna(row.get('ip_address')):
                ip = f"IP_{row['ip_address']}"
                self.nx_graph.add_node(ip, type='ip')
                self.nx_graph.add_edge(from_acc, ip, key=f"{tx_id}_ip", type='LOGGED_IN_FROM', timestamp=row['timestamp'])
                
            if pd.notna(row.get('atm_id')):
                atm = f"ATM_{row['atm_id']}"
                self.nx_graph.add_node(atm, type='atm')
                self.nx_graph.add_edge(from_acc, atm, key=f"{tx_id}_atm", type='WITHDREW_AT', amount=row['amount'], timestamp=row['timestamp'])
        
        return self.nx_graph

    def update_graph(self, tx_dict):
        """Real-time update logic from a single transaction dictionary."""
        tx_id = tx_dict['transaction_id']
        from_acc = f"ACC_{tx_dict['from_account']}"
        self.nx_graph.add_node(from_acc, type='account')
        
        if tx_dict.get('to_account'):
            to_acc = f"ACC_{tx_dict['to_account']}"
            self.nx_graph.add_node(to_acc, type='account')
            self.nx_graph.add_edge(from_acc, to_acc, key=tx_id, type='TRANSFERRED_TO', 
                                   amount=tx_dict['amount'], timestamp=tx_dict['timestamp'], channel=tx_dict['channel'])
        
        if tx_dict.get('device_id'):
            device = f"DEV_{tx_dict['device_id']}"
            self.nx_graph.add_node(device, type='device')
            self.nx_graph.add_edge(from_acc, device, key=f"{tx_id}_dev", type='USED_DEVICE', timestamp=tx_dict['timestamp'])
            
        if tx_dict.get('ip_address'):
            ip = f"IP_{tx_dict['ip_address']}"
            self.nx_graph.add_node(ip, type='ip')
            self.nx_graph.add_edge(from_acc, ip, key=f"{tx_id}_ip", type='LOGGED_IN_FROM', timestamp=tx_dict['timestamp'])
            
        if tx_dict.get('atm_id'):
            atm = f"ATM_{tx_dict['atm_id']}"
            self.nx_graph.add_node(atm, type='atm')
            self.nx_graph.add_edge(from_acc, atm, key=f"{tx_id}_atm", type='WITHDREW_AT', amount=tx_dict['amount'], timestamp=tx_dict['timestamp'])
            
        return self.nx_graph

    def get_pyg_heterodata(self, df):
        """Converts to PyTorch Geometric HeteroData format for GNN."""
        # A simplified mapping - in a real scenario you'd map IDs to integers and encode features properly.
        # This function provides a structural stub for HeteroData generation.
        data = HeteroData()
        # Extract accounts, devices, etc to create tensors.
        # For this prototype we will mainly use NetworkX features to feed a simpler classifier or a homogenous PyG wrap.
        pass

if __name__ == "__main__":
    import os
    if os.path.exists("../dataset/synthetic_transactions.csv"):
        df = pd.read_csv("../dataset/synthetic_transactions.csv")
        gb = GraphBuilder()
        g = gb.build_from_dataframe(df.head(100))
        print(f"Graph initialized with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
