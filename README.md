# 51% Attack Simulation - Blockchain Security Demo

Interactive web-based demonstration of blockchain attacks and defenses from the attacker's perspective.

## Quick Start

### Install Dependencies
```bash
pip install flask flask-cors
```

### Run Web Server
```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

## Understanding the Attack

### The Problem: 51% Attack
An attacker (Eve) with majority hash power can:
1. **Crack weak RSA keys** - Factor small primes to steal private keys
2. **Mine a longer secret chain** - Create blocks faster than honest miners
3. **Double spend** - Broadcast the longer chain to reverse transactions

### Attack Sequence
1. **Crack RSA Key** → Factor Alice's public key (n=3233) to get private key
2. **Launch 51% Attack** → Mine 3 consecutive blocks with fraudulent transaction
3. **Broadcast** → If chain is longer, network accepts it (vulnerability)

## Understanding the Defense

### Defense 1: Consecutive Block Limit (CBL) - Basic
- **Rule**: No miner can mine more than 2 consecutive blocks
- **Effect**: Blocks Eve's 51% attack (she mines 3 blocks)
- **Limitation**: Can be bypassed with Sybil attack (alternating miners)

### Defense 2: ECC + Stake-Weighted CBL
- **ECC Cryptography**: Prevents key theft (computationally secure)
- **Stake-Weighted CBL**: Validates chain by economic weight, not just length
- **Slashing**: Attackers lose stake when caught
- **Effect**: Blocks both key theft and Sybil attacks

## How to Use the Web Interface

### Left Panel (Attacker - Eve)
- **Crack RSA Key**: Factor victim's public key
- **51% Attack**: Mine longer chain with double spend
- **Double Spend (Bob)**: Attack different victim
- **Sybil Attack**: Use multiple identities to bypass CBL

### Right Panel (Network Defense)
- **Enable CBL (Basic)**: Activate Consecutive Block Limit defense
- **Enable ECC + Stake**: Upgrade to full security with slashing

### Center Panel
- **Blockchain Visualization**: Interactive graph showing honest (blue) and attack (red) chains
- **Genesis Block**: Highlighted with special border

### Demo Flow
1. **Start**: Network is vulnerable (no defenses)
2. **Attack**: Click "Crack RSA" → "51% Attack" → Attack succeeds
3. **Basic Defense**: Click "Enable CBL" → Try attack again → 51% blocked, but Sybil works
4. **Full Defense**: Click "Enable ECC + Stake" → Try "Sybil Attack" → Blocked with slashing

## Key Concepts

- **RSA Vulnerability**: Small primes (p=61, q=53) can be factored quickly
- **51% Attack**: Mining longer chain allows double spending
- **CBL Defense**: Limits consecutive blocks per miner (catches 51% but not Sybil)
- **Sybil Attack**: Multiple identities bypass basic CBL
- **Stake-Weighted CBL**: Economic weight prevents Sybil attacks
- **Stake Slashing**: Attackers lose stake when caught
- **ECC Security**: Elliptic curve cryptography prevents key theft

## Files

- `app.py` - Flask web server backend
- `templates/index.html` - Web interface
- `static/css/style.css` - Styling
- `static/js/app.js` - Frontend logic and visualization
- `51%ATTACK.md` - Complete software specification and architecture
- `requirements.txt` - Python dependencies
- `start_server.bat` / `start_server.sh` - Server startup scripts

## Requirements

- Python 3.7+
- Flask, flask-cors
- Modern web browser (Chrome, Firefox, Edge)

## Architecture

- **Backend**: Flask REST API with OOP blockchain simulation
- **Frontend**: Modern HTML/CSS/JavaScript with vis.js for graph visualization
- **Real-time Updates**: Polling-based state synchronization
- **Interactive**: Click buttons to trigger attacks and defenses
