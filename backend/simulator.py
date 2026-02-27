import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

def generate_accounts(num=500):
    return [str(uuid.uuid4())[:8] for _ in range(num)]

def generate_devices(num=100):
    return [str(uuid.uuid4())[:8] for _ in range(num)]

def generate_ips(num=100):
    return [f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(num)]

def generate_atms(num=20):
    return [f"ATM_{i}" for i in range(num)]

def simulate_normal_transactions(accounts, devices, ips, atms, num_tx=5000):
    transactions = []
    start_time = datetime.now() - timedelta(days=7)
    
    channels = ["APP", "WEB", "ATM", "UPI"]
    
    for _ in range(num_tx):
        tx_type = random.choice(channels)
        from_acc = random.choice(accounts)
        to_acc = random.choice(accounts) if tx_type != "ATM" else None
        if from_acc == to_acc: continue
            
        time = start_time + timedelta(minutes=random.randint(1, 10000))
        amount = round(random.uniform(10, 5000), 2)
        
        tx = {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": time,
            "from_account": from_acc,
            "to_account": to_acc,
            "channel": tx_type,
            "amount": amount,
            "device_id": random.choice(devices) if tx_type in ["APP", "WEB"] else None,
            "ip_address": random.choice(ips) if tx_type in ["APP", "WEB"] else None,
            "atm_id": random.choice(atms) if tx_type == "ATM" else None,
            "is_fraud": 0,
            "fraud_pattern": None
        }
        transactions.append(tx)
        
    return transactions

def inject_fraud_patterns(transactions, accounts, devices, ips, atms):
    # Pattern 1: High Velocity Cash Out
    # APP receive 50,000 -> UPI transfer -> ATM
    for _ in range(20):
        victim = random.choice(accounts)
        mule1 = random.choice(accounts)
        mule2 = random.choice(accounts)
        time = datetime.now() - timedelta(days=random.randint(1, 6))
        
        tx1 = {
            "transaction_id": str(uuid.uuid4()), "timestamp": time, "from_account": victim, "to_account": mule1,
            "channel": "APP", "amount": 50000, "device_id": random.choice(devices), "ip_address": random.choice(ips),
            "atm_id": None, "is_fraud": 1, "fraud_pattern": "High Velocity Cash Out"
        }
        time2 = time + timedelta(minutes=random.randint(1, 2))
        tx2 = {
            "transaction_id": str(uuid.uuid4()), "timestamp": time2, "from_account": mule1, "to_account": mule2,
            "channel": "UPI", "amount": 49900, "device_id": None, "ip_address": None,
            "atm_id": None, "is_fraud": 1, "fraud_pattern": "High Velocity Cash Out"
        }
        time3 = time2 + timedelta(minutes=random.randint(1, 2))
        tx3 = {
            "transaction_id": str(uuid.uuid4()), "timestamp": time3, "from_account": mule2, "to_account": None,
            "channel": "ATM", "amount": 49500, "device_id": None, "ip_address": None,
            "atm_id": random.choice(atms), "is_fraud": 1, "fraud_pattern": "High Velocity Cash Out"
        }
        transactions.extend([tx1, tx2, tx3])

    # Pattern 2: Layered Mule Ring
    for _ in range(15):
        mules = random.sample(accounts, 4)
        time = datetime.now() - timedelta(days=random.randint(1, 6))
        amount = 100000
        for i in range(3):
            tx = {
                "transaction_id": str(uuid.uuid4()), "timestamp": time, "from_account": mules[i], "to_account": mules[i+1],
                "channel": "UPI", "amount": amount, "device_id": random.choice(devices), "ip_address": random.choice(ips),
                "atm_id": None, "is_fraud": 1, "fraud_pattern": "Layered Mule Ring"
            }
            time += timedelta(minutes=random.randint(30, 120))
            amount *= random.uniform(0.9, 0.98) # Slight drop
            transactions.append(tx)
        
        tx_atm = {
            "transaction_id": str(uuid.uuid4()), "timestamp": time, "from_account": mules[3], "to_account": None,
            "channel": "ATM", "amount": amount*0.95, "device_id": None, "ip_address": None,
            "atm_id": random.choice(atms), "is_fraud": 1, "fraud_pattern": "Layered Mule Ring"
        }
        transactions.append(tx_atm)

    # Pattern 3: Circular Movement
    for _ in range(10):
        mules = random.sample(accounts, 4)
        time = datetime.now() - timedelta(days=random.randint(1, 6))
        amount = 50000
        for i in range(4):
            to_acc = mules[(i+1)%4]
            tx = {
                "transaction_id": str(uuid.uuid4()), "timestamp": time, "from_account": mules[i], "to_account": to_acc,
                "channel": "WEB", "amount": amount, "device_id": random.choice(devices), "ip_address": random.choice(ips),
                "atm_id": None, "is_fraud": 1, "fraud_pattern": "Circular Movement"
            }
            time += timedelta(minutes=random.randint(5, 30))
            amount *= random.uniform(0.95, 1.0)
            transactions.append(tx)

    # Pattern 4: Device Reuse (Multiple accounts same device/IP)
    for _ in range(5):
        shared_device = random.choice(devices)
        shared_ip = random.choice(ips)
        mules = random.sample(accounts, 5)
        time = datetime.now() - timedelta(days=random.randint(1, 6))
        for mule in mules:
            tx = {
                "transaction_id": str(uuid.uuid4()), "timestamp": time, "from_account": mule, "to_account": random.choice(accounts),
                "channel": "APP", "amount": random.uniform(5000, 20000), "device_id": shared_device, "ip_address": shared_ip,
                "atm_id": None, "is_fraud": 1, "fraud_pattern": "Device/IP Reuse"
            }
            time += timedelta(minutes=random.randint(1, 10))
            transactions.append(tx)

    return transactions

def generate_dataset():
    accounts = generate_accounts(1000)
    devices = generate_devices(200)
    ips = generate_ips(300)
    atms = generate_atms(50)
    
    transactions = simulate_normal_transactions(accounts, devices, ips, atms, 15000)
    transactions = inject_fraud_patterns(transactions, accounts, devices, ips, atms)
    
    df = pd.DataFrame(transactions)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    import os
    os.makedirs("c:/Users/jeeva/Desktop/fraud_detection/MuleNet/dataset", exist_ok=True)
    df.to_csv("c:/Users/jeeva/Desktop/fraud_detection/MuleNet/dataset/synthetic_transactions.csv", index=False)
    print(f"Generated {len(df)} transactions. Dataset saved.")
    return df

if __name__ == "__main__":
    generate_dataset()
