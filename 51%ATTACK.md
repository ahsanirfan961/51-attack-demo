# Software Specification: Blockchain 51% Attack & Defense Simulator

**Project Title:** Simulating Consecutive Block Limits (CBL) and Stake-Governance as Defenses Against 51% Attacks

**Base Research:** Babur et al., "Preventing 51% Attack By Using Consecutive Block Limits In Bitcoin" (IEEE Access, 2024).

---

## 1. System Overview

The system is a Python-based interactive simulation of a Proof-of-Work (PoW) blockchain. It visualizes the "Arms Race" between a malicious actor (Eve) and the Consensus Protocol. The simulation is state-based, meaning it accurately tracks wallet balances (UTXOs), cryptographic signatures, and chain validity. It does not use "fake" probabilities; it uses actual data structures and cryptographic logic.

### 1.1 Goals

1. Demonstrate a Double Spend attack using 51% hashrate and RSA key cracking.
2. Implement Static CBL (from the paper) to block the basic attack.
3. Demonstrate a Sybil Attack where the attacker creates multiple identities to bypass Static CBL.
4. Implement Stake-Weighted CBL (Stake-CBL) and ECC to permanently secure the network.

---

## 2. System Actors & Assets

### 2.1 The Entities

**Alice (The Honest Victim):**
- **Role:** Honest Miner & User.
- **Assets:** High Stake (e.g., 5,000 Coins), Initial Balance (100 Coins).
- **Behavior:** Mines on the longest valid chain. Broadcasts valid transactions.

**Bob (The Honest Miner):**
- **Role:** Supports Alice. Mines on the honest chain to add legitimate weight.

**Eve (The Attacker):**
- **Role:** Malicious Miner.
- **Capabilities:** Can switch hashrate (0% to 51%), can run factorization algorithms, can spawn "Sybil" sub-wallets.
- **Assets:** Low Stake (e.g., 200 Coins), Low Initial Balance.

### 2.2 The Economy (Conservation of Funds)

**The Ledger:** The "Truth" is defined solely by the currently active blockchain tip.

**Fund Restoration Logic:** If an attack chain is rejected, the ledger state reverts to the Honest Chain. This ensures Alice's funds are "restored" simply because the theft transaction never effectively happened in the valid history.

---

## 3. Functional Architecture

The system consists of three distinct "Engines" that interact interactively.

### 3.1 The Cryptographic Engine

**Mode A: Weak RSA (Legacy):**
- Generates keys using small primes ($p, q < 100$).
- **Vulnerability:** Allows Eve to run `factorize(n)` to derive the private key $d$.

**Mode B: ECC (Modern):**
- Uses Elliptic Curve logic (simulated secp256k1).
- **Security:** Factorization is impossible. Eve cannot sign transactions on behalf of Alice.

### 3.2 The Mining Engine (The Graph)

- **Structure:** Directed Acyclic Graph (DAG).
- **Block Data:** Index, Prev_Hash, Miner_ID, Transactions, Timestamp.
- **Hashing:** SHA-256. Any change in transaction data alters the block hash, breaking the chain.

### 3.3 The Consensus Engine (The Rules)

This is the core logic that changes as the simulation upgrades.

- **Level 1 (Nakamoto):** Valid Chain = Longest Chain.
- **Level 2 (Static CBL):** Valid Chain = Longest Chain AND No miner mines $>N$ consecutive blocks.
- **Level 3 (Stake-CBL):** Valid Chain = Heaviest Stake Weight AND Dynamic limits based on Stake.

---

## 4. Simulation Scenarios (The Story Arc)

### Scenario 1: The "RSA Heist" & Basic 51% Attack

**State:** Legacy Network.

1. **Attacker Action:** Eve runs `brute_force_rsa(Alice_Pub_Key)`.
2. **Network State:** Eve obtains Alice's Private Key.
3. **Attacker Action:** Eve creates a TX: `Alice -> Eve (100 Coins)`.
4. **Attacker Action:** Eve engages "High Power Mode" (51% Hashrate). She mines a secret fork of 3 blocks containing the theft.
5. **Network Response (Vulnerable):** Eve broadcasts the chain. The Consensus Engine sees `Len(Eve) > Len(Honest)`.
6. **Outcome:** Reorg occurs. The Honest Chain is orphaned. Alice's balance becomes 0.

### Scenario 2: The "Paper" Defense (Static CBL)

**State:** CBL Protocol Activated ($N=2$).

1. **Attacker Action:** Eve repeats the exact attack from Scenario 1. She mines 3 blocks consecutively.
2. **Network Response (Secure):** The Consensus Engine scans the new chain. It detects Eve appears 3 times in a row.
3. **Outcome:** The chain is Rejected as invalid.
4. **Fund Status:** The network stays on the Honest Chain. Alice's balance remains 100. Eve wastes electricity.

### Scenario 3: The "Sybil" Adaptation

**State:** CBL Protocol Active ($N=2$).

1. **Attacker Logic:** Eve realizes the check is on Miner_ID.
2. **Attacker Action:** Eve creates Eve_A and Eve_B. She splits her stake/resources between them.
3. **Attacker Action:** She mines a secret chain: `Block1(Eve_A) -> Block2(Eve_B) -> Block3(Eve_A)`.
4. **Network Response (Vulnerable):** The Consensus Engine checks consecutives. `Eve_A != Eve_B`. The check passes.
5. **Outcome:** Reorg occurs. Eve bypasses the CBL defense. Alice loses funds.

### Scenario 4: The Final Defense (Stake-CBL + ECC)

**State:** Governance Protocol Active.

1. **Crypto Upgrade:** Network moves to ECC. Eve tries to crack Alice's key but fails. She cannot steal Alice's funds, so she tries to double-spend her own funds (Theft of service/goods).
2. **Attacker Action:** Eve tries the Sybil Attack again to force a reorg.
3. **Network Response (Secure):**
   - **Weight Check:** The Consensus Engine calculates Chain Weight.
   - **Honest:** Alice (5000 Stake) + Bob (5000 Stake) = Weight 10,000.
   - **Attacker:** Eve_A (100 Stake) + Eve_B (100 Stake) = Weight 200.
4. **Outcome:** `Attacker_Weight < Honest_Weight`. The chain is rejected.
5. **Punishment:** The "Governance" module logs the attempt and (in a real system) would slash Eve's stake.

---

## 5. User Interface (Dashboard) Specs

We use a web-based interface with Flask backend and JavaScript frontend.

### 5.1 The Visuals

**The DAG Plot:**
- **Blue Nodes:** The Honest Chain (Mainnet).
- **Red Nodes:** The Attack Chain (Fork).
- **Visual Logic:** When a Reorg happens, the Red chain moves to the top (becomes Mainnet) and Blue becomes orphaned (greyed out). When an attack is blocked, the Red chain simply terminates/disappears.

**Wallet Bar Chart:** Real-time bars showing Alice, Bob, and Eve balances. This proves the "Double Spend" visually (Alice's bar drops to zero).

### 5.2 The Controls

- **Crack Keys Button:** Triggers the RSA Engine.
- **Mine Attack Block Button:** Adds a red block to the secret chain.
- **Broadcast Chain Button:** Triggers the Consensus Engine to judge the fork.
- **Protocol Toggle:** Switch between "Legacy", "CBL", and "Stake-Gov".

---

## 6. Implementation Logic (Pseudo-Code)

### 6.1 The Fund Restoration Logic

This is how we ensure the simulation is "Authentic":

```python
def resolve_fork(current_chain, new_chain):
    # 1. Validate Rules
    if not consensus_engine.validate(new_chain):
        return # Reject. Current chain stays. Funds unchanged.
    
    # 2. If Accepted, Re-calculate State
    # We do NOT just edit balances. We re-run history.
    
    # Wipe current balances (Logic only, not data loss)
    reset_balances_to_genesis()
    
    # Re-process transactions from the NEW chain
    for block in new_chain:
        apply_transactions(block.txs)
        
    # Result: If the new chain has "Alice->Eve", Eve gets money.
    # If we revert to old chain, we re-process "Alice->Bob", Alice gets money.
```

This specification ensures that the simulation is not just an animation, but a functioning logical model of a blockchain ledger under siege.
