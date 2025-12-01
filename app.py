#!/usr/bin/env python3
"""
51% Attack Simulation - Strict Implementation
Based on: 51%ATTACK.md Specification
Demonstrates RSA cracking, 51% attacks, CBL defense, Sybil attacks, and Stake-CBL defense
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import hashlib
import time
import math
from typing import Dict, List, Tuple, Optional

app = Flask(__name__)
CORS(app)

# ==========================================
# CRYPTOGRAPHIC ENGINE
# ==========================================

class CryptoEngine:
    """Handles RSA key generation and cracking (weak RSA) and ECC simulation"""
    
    @staticmethod
    def mod_inverse(a, m):
        """Calculate modular inverse using extended Euclidean algorithm"""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            return None
        return (x % m + m) % m
    
    @staticmethod
    def generate_weak_rsa():
        """Generate weak RSA keys with small primes (p=61, q=53) for demonstration"""
        p, q = 61, 53  # Small primes - vulnerable!
        n = p * q  # 3233
        phi = (p - 1) * (q - 1)  # 3120
        e = 17  # Public exponent
        d = CryptoEngine.mod_inverse(e, phi)  # Private exponent
        return ((e, n), (d, n))  # (public_key, private_key)
    
    @staticmethod
    def crack_rsa(public_key):
        """Factor RSA modulus n to crack private key"""
        e, n = public_key
        # Brute force factorization (fast for small primes)
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                p = i
                q = n // i
                phi = (p - 1) * (q - 1)
                d = CryptoEngine.mod_inverse(e, phi)
                return (d, n)  # Return private key
        return None
    
    @staticmethod
    def generate_ecc():
        """Simulate ECC keys (uncrackable)"""
        # In real ECC, this would be a point on elliptic curve secp256k1
        # For simulation, we mark it as secure
        return ("ECC_SECURE", "ECC_SECURE")

# ==========================================
# BLOCKCHAIN STRUCTURE
# ==========================================

class Wallet:
    """Represents a blockchain participant"""
    def __init__(self, name: str, balance: float, stake: int, is_ecc: bool = False):
        self.name = name
        self.balance = balance
        self.original_balance = balance
        self.stake = stake
        self.original_stake = stake
        self.is_ecc = is_ecc
        self.is_compromised = False
        
        if is_ecc:
            self.pub_key, self.priv_key = CryptoEngine.generate_ecc()
        else:
            self.pub_key, self.priv_key = CryptoEngine.generate_weak_rsa()
    
    def to_dict(self):
        return {
            'name': self.name,
            'balance': self.balance,
            'original_balance': self.original_balance,
            'stake': self.stake,
            'original_stake': self.original_stake,
            'is_ecc': self.is_ecc,
            'is_compromised': self.is_compromised,
            'pub_key': str(self.pub_key) if isinstance(self.pub_key, tuple) else self.pub_key
        }

class Transaction:
    """Represents a blockchain transaction"""
    def __init__(self, tx_id: str, from_addr: str, to_addr: str, amount: float, signature: str):
        self.tx_id = tx_id
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.amount = amount
        self.signature = signature
        self.timestamp = time.time()
    
    def to_dict(self):
        return {
            'tx_id': self.tx_id,
            'from_addr': self.from_addr,
            'to_addr': self.to_addr,
            'amount': self.amount,
            'signature': self.signature[:16] + '...'
        }

class Block:
    """Represents a blockchain block"""
    def __init__(self, index: int, prev_hash: str, miner: str, transactions: List[Transaction]):
        self.index = index
        self.prev_hash = prev_hash
        self.miner = miner
        self.transactions = transactions
        self.timestamp = time.time()
        
        # Calculate block hash
        tx_data = ''.join([f"{tx.from_addr}{tx.to_addr}{tx.amount}" for tx in transactions])
        block_data = f"{index}{prev_hash}{miner}{tx_data}{self.timestamp}"
        self.hash = hashlib.sha256(block_data.encode()).hexdigest()
    
    def to_dict(self):
        return {
            'index': self.index,
            'hash': self.hash[:16] + '...',
            'prev_hash': self.prev_hash[:16] + '...' if len(self.prev_hash) > 16 else self.prev_hash,
            'miner': self.miner,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'timestamp': self.timestamp
        }

class Blockchain:
    """Represents a blockchain ledger"""
    def __init__(self, name: str = "Chain"):
        self.name = name
        genesis = Block(0, "0" * 64, "Genesis", [])
        self.chain = [genesis]
    
    def add_block(self, miner: str, transactions: List[Transaction]) -> Block:
        """Add a new block to the chain"""
        prev_hash = self.chain[-1].hash
        new_block = Block(len(self.chain), prev_hash, miner, transactions)
        self.chain.append(new_block)
        return new_block
    
    def to_dict(self):
        return [block.to_dict() for block in self.chain]
    
    def calculate_balance(self, address: str, wallets: Dict[str, Wallet]) -> float:
        """Calculate balance by replaying all transactions"""
        balance = wallets[address].original_balance
        for block in self.chain[1:]:  # Skip genesis
            for tx in block.transactions:
                if tx.from_addr == address:
                    balance -= tx.amount
                if tx.to_addr == address:
                    balance += tx.amount
        return balance

# ==========================================
# EXECUTION TRACKER
# ==========================================

class ExecutionTracker:
    """Tracks code execution steps for visualization"""
    
    def __init__(self):
        self.steps = []
        self.current_function = None
        self.call_stack = []
    
    def add_step(self, step: str, status: str = "checking", details: str = "", function: str = None, 
                 code_snippet: str = "", security_context: str = ""):
        """Add an execution step with code snippet and security context"""
        if function:
            self.current_function = function
        
        step_data = {
            'step': step,
            'status': status,  # "checking", "passed", "failed"
            'details': details,
            'code_snippet': code_snippet,
            'security_context': security_context,
            'timestamp': time.time(),
            'function': function or self.current_function
        }
        self.steps.append(step_data)
        return step_data
    
    def push_function(self, function_name: str):
        """Push function onto call stack"""
        self.call_stack.append(function_name)
        self.current_function = function_name
    
    def pop_function(self):
        """Pop function from call stack"""
        if self.call_stack:
            self.call_stack.pop()
            self.current_function = self.call_stack[-1] if self.call_stack else None
    
    def clear(self):
        """Clear all execution steps"""
        self.steps = []
        self.current_function = None
        self.call_stack = []
    
    def to_dict(self):
        """Convert to dictionary for JSON response"""
        return {
            'steps': self.steps[-50:],  # Last 50 steps
            'current_function': self.current_function,
            'call_stack': self.call_stack.copy()
        }

# ==========================================
# CONSENSUS ENGINE
# ==========================================

class ConsensusEngine:
    """Validates chains according to consensus rules"""
    
    @staticmethod
    def validate_pow(chain: List[Block]) -> Tuple[bool, str]:
        """Validate Proof of Work - hash chain integrity"""
        for i in range(1, len(chain)):
            if chain[i].prev_hash != chain[i-1].hash:
                return False, f"PoW violation: Hash mismatch at block {i}"
        return True, "PoW valid"
    
    @staticmethod
    def check_static_cbl(chain: List[Block], max_consecutive: int = 2) -> Tuple[bool, str]:
        """Check Static CBL - no miner can mine more than N consecutive blocks"""
        if len(chain) <= 1:
            return True, ""
        
        # Skip genesis, start from first real block
        consecutive = 1
        last_miner = chain[1].miner if len(chain) > 1 else None
        
        # Check from block 2 onwards
        for i in range(2, len(chain)):
            block = chain[i]
            if block.miner == last_miner:
                consecutive += 1
                if consecutive > max_consecutive:
                    return False, f"CBL violation: {block.miner} mined {consecutive} consecutive blocks (limit: {max_consecutive})"
            else:
                consecutive = 1
                last_miner = block.miner
        
        return True, ""
    
    @staticmethod
    def check_stake_weight(chain: List[Block], wallets: Dict[str, Wallet]) -> int:
        """Calculate total stake weight of a chain"""
        total_stake = 0
        for block in chain[1:]:  # Skip genesis
            miner_wallet = wallets.get(block.miner)
            if miner_wallet:
                total_stake += miner_wallet.stake
            # Handle Sybil identities (Eve_A, Eve_B)
            elif block.miner.startswith("Eve_"):
                eve_wallet = wallets.get("Eve")
                if eve_wallet:
                    total_stake += eve_wallet.stake // 2  # Split Eve's stake
        return total_stake
    
    @staticmethod
    def validate_fork(attack_chain: List[Block], honest_chain: List[Block],
                     defense_mode: str, wallets: Dict[str, Wallet], 
                     execution_tracker: Optional['ExecutionTracker'] = None) -> Tuple[bool, str, List[str]]:
        """
        Validate attack chain against honest chain
        Returns: (is_valid, reason, slashed_miners)
        """
        if execution_tracker:
            execution_tracker.push_function("ConsensusEngine.validate_fork")
            execution_tracker.add_step("Start validation", "checking", f"Attack chain: {len(attack_chain)} blocks, Honest: {len(honest_chain)} blocks")
        
        slashed_miners = []
        
        # 1. PoW Validation
        if execution_tracker:
            execution_tracker.add_step(
                "PoW Validation", 
                "checking", 
                "Checking hash chain integrity",
                code_snippet="class ConsensusEngine:\n    @staticmethod\n    def validate_pow(chain):\n        for i in range(1, len(chain)):\n            if chain[i].prev_hash != chain[i-1].hash:\n                return False\n        return True",
                security_context="OOP: ConsensusEngine class validates chain integrity (PoW)"
            )
        pow_valid, pow_reason = ConsensusEngine.validate_pow(attack_chain)
        if not pow_valid:
            if execution_tracker:
                execution_tracker.add_step("PoW Validation", "failed", pow_reason)
                execution_tracker.pop_function()
            return False, pow_reason, slashed_miners
        if execution_tracker:
            execution_tracker.add_step(
                "PoW Validation", 
                "passed", 
                "Hash chain is valid",
                code_snippet="# All prev_hash links match\nreturn True",
                security_context="Security: Chain integrity verified"
            )
        
        # 2. Defense Mode Checks
        if execution_tracker:
            execution_tracker.add_step("Check Defense Mode", "checking", f"Mode: {defense_mode}")
        
        if defense_mode == "LEGACY":
            # Level 1: Nakamoto - Longest chain wins
            if execution_tracker:
                execution_tracker.add_step("Defense Mode: LEGACY", "checking", "Using longest chain rule")
            if len(attack_chain) > len(honest_chain):
                if execution_tracker:
                    execution_tracker.add_step("Check Chain Length", "passed", f"Attack ({len(attack_chain)}) > Honest ({len(honest_chain)})")
                    execution_tracker.add_step("Result: ACCEPTED", "passed", "Longest chain accepted")
                    execution_tracker.pop_function()
                return True, "Longest chain accepted", slashed_miners
            if execution_tracker:
                execution_tracker.add_step("Check Chain Length", "failed", f"Attack ({len(attack_chain)}) <= Honest ({len(honest_chain)})")
                execution_tracker.add_step("Result: REJECTED", "failed", "Attack chain too short")
                execution_tracker.pop_function()
            return False, "Attack chain too short", slashed_miners
        
        elif defense_mode == "CBL":
            # Level 2: Static CBL
            if execution_tracker:
                execution_tracker.add_step(
                    "Defense Mode: CBL", 
                    "checking", 
                    "Checking consecutive block limit",
                    code_snippet="class ConsensusEngine:\n    @staticmethod\n    def check_static_cbl(chain, max_consecutive=2):\n        consecutive = 1\n        for block in chain:\n            if block.miner == last_miner:\n                consecutive += 1\n                if consecutive > max_consecutive:\n                    return False",
                    security_context="OOP: ConsensusEngine.check_static_cbl() - CBL defense method"
                )
            cbl_valid, cbl_reason = ConsensusEngine.check_static_cbl(attack_chain, max_consecutive=2)
            if not cbl_valid:
                if execution_tracker:
                    execution_tracker.add_step(
                        "CBL Check", 
                        "failed", 
                        cbl_reason,
                        code_snippet="if consecutive > max_consecutive:\n    return False  # CBL violation",
                        security_context="Security: Attack blocked by CBL rule"
                    )
                    execution_tracker.add_step("Result: REJECTED", "failed", "CBL violation detected")
                    execution_tracker.pop_function()
                return False, cbl_reason, slashed_miners
            if execution_tracker:
                execution_tracker.add_step(
                    "CBL Check", 
                    "passed", 
                    "No consecutive block violation",
                    code_snippet="return True  # CBL check passed",
                    security_context="Defense: CBL rule satisfied"
                )
            
            # Still need to be longer
            if execution_tracker:
                execution_tracker.add_step("Check Chain Length", "checking", f"Attack: {len(attack_chain)}, Honest: {len(honest_chain)}")
            if len(attack_chain) > len(honest_chain):
                if execution_tracker:
                    execution_tracker.add_step("Check Chain Length", "passed", f"Attack ({len(attack_chain)}) > Honest ({len(honest_chain)})")
                    execution_tracker.add_step("Result: ACCEPTED", "passed", "Longest chain accepted (CBL passed)")
                    execution_tracker.pop_function()
                return True, "Longest chain accepted (CBL passed)", slashed_miners
            if execution_tracker:
                execution_tracker.add_step("Check Chain Length", "failed", f"Attack ({len(attack_chain)}) <= Honest ({len(honest_chain)})")
                execution_tracker.add_step("Result: REJECTED", "failed", "Attack chain too short")
                execution_tracker.pop_function()
            return False, "Attack chain too short", slashed_miners
        
        elif defense_mode == "STAKE_CBL":
            # Level 3: Stake-Weighted CBL
            if execution_tracker:
                execution_tracker.add_step(
                    "Defense Mode: STAKE_CBL", 
                    "checking", 
                    "Checking stake weight",
                    code_snippet="class ConsensusEngine:\n    @staticmethod\n    def check_stake_weight(chain, wallets):\n        weight = sum(wallets[block.miner].stake\n                     for block in chain)\n        return weight",
                    security_context="OOP: ConsensusEngine.check_stake_weight() - stake calculation method"
                )
            # Calculate stake weight of honest chain (miners who mined blocks)
            honest_weight = ConsensusEngine.check_stake_weight(honest_chain, wallets)
            attack_weight = ConsensusEngine.check_stake_weight(attack_chain, wallets)
            
            # Add honest stakeholder backing (Alice + Bob validate honest chain)
            # This represents economic weight backing the honest chain
            alice_stake = wallets.get('Alice', Wallet("Alice", 0, 0)).stake
            bob_stake = wallets.get('Bob', Wallet("Bob", 0, 0)).stake
            honest_backing = alice_stake + bob_stake
            honest_weight += honest_backing
            
            if execution_tracker:
                execution_tracker.add_step(
                    "Calculate Stake Weights", 
                    "checking", 
                    f"Attack: {attack_weight}, Honest: {honest_weight - honest_backing} + backing: {honest_backing} = {honest_weight}",
                    code_snippet="honest_weight = miner_stakes + alice_stake + bob_stake\nattack_weight = sum(eve_stakes)",
                    security_context="Consensus: Economic weight calculation"
                )
            
            if attack_weight < honest_weight:
                # Slash attacker's stake
                if execution_tracker:
                    execution_tracker.add_step(
                        "Stake Weight Check", 
                        "failed", 
                        f"Attack ({attack_weight}) < Honest ({honest_weight})",
                        code_snippet="if attack_weight < honest_weight:\n    return False  # Insufficient stake",
                        security_context="Security: Attack rejected - insufficient economic weight"
                    )
                    execution_tracker.add_step(
                        "Slash Attacker Stake", 
                        "checking", 
                        "Punishing attacker",
                        code_snippet="wallet.stake = max(0, wallet.stake - penalty)\n# Slashing mechanism",
                        security_context="Defense: Slashing punishes attackers"
                    )
                for block in attack_chain[1:]:
                    miner = block.miner
                    if miner.startswith("Eve") and miner not in slashed_miners:
                        slashed_miners.append(miner)
                        if miner == "Eve":
                            wallets["Eve"].stake = max(0, wallets["Eve"].stake - 50)
                        elif miner.startswith("Eve_"):
                            wallets["Eve"].stake = max(0, wallets["Eve"].stake - 25)
                if execution_tracker:
                    execution_tracker.add_step("Slash Attacker Stake", "passed", f"Slashed: {', '.join(slashed_miners)}")
                    execution_tracker.add_step("Result: REJECTED", "failed", 
                        f"Insufficient stake weight: Attack={attack_weight} < Honest={honest_weight}")
                    execution_tracker.pop_function()
                return False, f"Insufficient stake weight: Attack={attack_weight}, Honest={honest_weight} (including {honest_backing} stakeholder backing)", slashed_miners
            
            if execution_tracker:
                execution_tracker.add_step(
                    "Stake Weight Check", 
                    "passed", 
                    f"Attack ({attack_weight}) >= Honest ({honest_weight})",
                    code_snippet="return True  # Stake weight sufficient",
                    security_context="Consensus: Chain accepted by stake weight"
                )
                execution_tracker.add_step("Result: ACCEPTED", "passed", "Stake weight sufficient")
                execution_tracker.pop_function()
            return True, f"Stake weight sufficient: Attack={attack_weight} >= Honest={honest_weight}", slashed_miners
        
        if execution_tracker:
            execution_tracker.add_step("Result: REJECTED", "failed", "Unknown defense mode")
            execution_tracker.pop_function()
        return False, "Unknown defense mode", slashed_miners

# ==========================================
# SIMULATION STATE
# ==========================================

class SimulationState:
    """Manages the complete simulation state"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset to initial state"""
        self.wallets = {
            "Alice": Wallet("Alice", 100.0, 5000, is_ecc=False),
            "Bob": Wallet("Bob", 50.0, 5000, is_ecc=False),
            "Eve": Wallet("Eve", 10.0, 200, is_ecc=False)
        }
        
        # Initialize honest chain with one transaction
        self.honest_chain = Blockchain("HonestChain")
        alice_to_bob_tx = Transaction("tx_honest_1", "Alice", "Bob", 10.0, "alice_sig_1")
        self.honest_chain.add_block("Miner1", [alice_to_bob_tx])
        
        self.attack_chain = []
        self.defense_mode = "LEGACY"  # LEGACY, CBL, STAKE_CBL
        self.alice_cracked = False
        self.use_sybil = False  # Flag to indicate Sybil attack (Scenario 3)
        self.execution_tracker = ExecutionTracker()
        self.hash_power_distribution = {
            "Honest": 50.0,
            "Eve": 0.0
        }
        self.logs = [
            "=== BLOCKCHAIN 51% ATTACK SIMULATION ===",
            "Network: LEGACY (Vulnerable)",
            "Alice: 100 BTC, Stake: 5000",
            "Bob: 50 BTC, Stake: 5000",
            "Eve: 10 BTC, Stake: 200",
            "Ready to begin attack..."
        ]
        
        # Clear execution tracker
        self.execution_tracker.clear()
        
        # Recalculate balances from honest chain
        for wallet in self.wallets.values():
            wallet.balance = wallet.original_balance
        
        for block in self.honest_chain.chain[1:]:
            for tx in block.transactions:
                if tx.from_addr in self.wallets:
                    self.wallets[tx.from_addr].balance -= tx.amount
                if tx.to_addr in self.wallets:
                    self.wallets[tx.to_addr].balance += tx.amount
    
    def resolve_fork(self, attack_chain: List[Block]):
        """
        Resolve fork according to specification - re-calculate balances from chain
        This implements the fund restoration logic from 51%ATTACK.md
        """
        is_valid, reason, slashed = ConsensusEngine.validate_fork(
            attack_chain,
            self.honest_chain.chain,
            self.defense_mode,
            self.wallets,
            self.execution_tracker
        )
        
        if is_valid:
            # Attack chain accepted - re-calculate balances from new chain
            self.honest_chain.chain = attack_chain.copy()
            self.attack_chain = []
            self.execution_tracker.clear()  # Clear execution tracker after successful validation
            
            # Re-process transactions from new chain
            for wallet in self.wallets.values():
                wallet.balance = wallet.original_balance
            
            for block in self.honest_chain.chain[1:]:
                for tx in block.transactions:
                    if tx.from_addr in self.wallets:
                        self.wallets[tx.from_addr].balance -= tx.amount
                    if tx.to_addr in self.wallets:
                        self.wallets[tx.to_addr].balance += tx.amount
            
            return True, reason, slashed
        else:
            # Attack rejected - recalculate balances from honest chain
            # If CBL blocked the attack, enable Sybil for next attempt (Scenario 3)
            if self.defense_mode == "CBL" and "CBL violation" in reason:
                self.use_sybil = True
                self.logs.append("[EVE] Attack blocked by CBL! Switching to Sybil attack (Scenario 3)")
                self.logs.append("[EVE] Creating Eve_A and Eve_B identities to bypass CBL...")
            
            self.attack_chain = []
            
            # Re-process transactions from honest chain to ensure correct balances
            for wallet in self.wallets.values():
                wallet.balance = wallet.original_balance
            
            for block in self.honest_chain.chain[1:]:
                for tx in block.transactions:
                    if tx.from_addr in self.wallets:
                        self.wallets[tx.from_addr].balance -= tx.amount
                    if tx.to_addr in self.wallets:
                        self.wallets[tx.to_addr].balance += tx.amount
            
            return False, reason, slashed
    
    def to_dict(self):
        """Convert state to dictionary for JSON response"""
        return {
            'wallets': {k: v.to_dict() for k, v in self.wallets.items()},
            'honest_chain': self.honest_chain.to_dict(),
            'attack_chain': [b.to_dict() for b in self.attack_chain],
            'defense_mode': self.defense_mode,
            'alice_cracked': self.alice_cracked,
            'use_sybil': self.use_sybil,
            'hash_power': self.hash_power_distribution,
            'execution_tracker': self.execution_tracker.to_dict(),
            'logs': self.logs[-30:]  # Last 30 logs
        }

# Global simulation state
sim_state = SimulationState()

# ==========================================
# API ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(sim_state.to_dict())

@app.route('/api/crack_rsa', methods=['POST'])
def crack_rsa():
    """Scenario 1 Step 1: Crack Alice's RSA key"""
    sim_state.execution_tracker.clear()
    sim_state.execution_tracker.push_function("crack_rsa")
    sim_state.execution_tracker.add_step(
        "Start RSA Cracking", 
        "checking", 
        "Beginning key cracking process",
        code_snippet="class Wallet:\n    def __init__(self, name, balance, stake, is_ecc=False):\n        self.pub_key, self.priv_key = CryptoEngine.generate_weak_rsa()",
        security_context="OOP: Wallet class (Identity Layer) - manages cryptographic keys"
    )
    
    if sim_state.defense_mode == "STAKE_CBL":
        sim_state.execution_tracker.add_step(
            "Check Defense Mode", 
            "checking", 
            "STAKE_CBL (ECC enabled)",
            code_snippet="if wallet.is_ecc:\n    # ECC keys cannot be factorized\n    return None",
            security_context="OOP: Wallet.is_ecc property determines cryptography type"
        )
        sim_state.execution_tracker.add_step(
            "Attempt ECC Cracking", 
            "failed", 
            "ECC is computationally secure",
            code_snippet="# Elliptic Curve Discrete Logarithm Problem\n# No known polynomial-time solution",
            security_context="Cryptography: ECC provides quantum-resistant encryption"
        )
        sim_state.logs.append("[EVE] Attempting to crack ECC key...")
        sim_state.logs.append("[EVE] FAILED: ECC cryptography is computationally secure")
        sim_state.logs.append("[EVE] Cannot factorize elliptic curve discrete logarithm problem")
        sim_state.execution_tracker.pop_function()
        return jsonify({'success': False, 'message': 'ECC is uncrackable'})
    
    if sim_state.alice_cracked:
        sim_state.execution_tracker.add_step("Check Already Cracked", "failed", "Key already compromised")
        sim_state.execution_tracker.pop_function()
        return jsonify({'success': False, 'message': 'Alice\'s key already cracked'})
    
    sim_state.execution_tracker.add_step(
        "Check Defense Mode", 
        "passed", 
        "RSA (vulnerable)",
        code_snippet="wallet.is_ecc == False  # Using Weak RSA\n# Small primes allow factorization",
        security_context="OOP: Wallet property exposes vulnerability"
    )
    sim_state.execution_tracker.add_step(
        "Get Public Key", 
        "checking", 
        f"e={sim_state.wallets['Alice'].pub_key[0]}, n={sim_state.wallets['Alice'].pub_key[1]}",
        code_snippet="pub_key = wallet.pub_key  # (e, n)\n# Public key exposes modulus n = p * q",
        security_context="OOP: Wallet.pub_key property access"
    )
    sim_state.logs.append("[EVE] === CRACKING RSA KEY ===")
    sim_state.logs.append(f"[EVE] Alice's Public Key: e={sim_state.wallets['Alice'].pub_key[0]}, n={sim_state.wallets['Alice'].pub_key[1]}")
    sim_state.logs.append("[EVE] Running factorization algorithm...")
    
    sim_state.execution_tracker.add_step(
        "Factorize RSA Modulus", 
        "checking", 
        "Brute force factorization",
        code_snippet="class CryptoEngine:\n    @staticmethod\n    def crack_rsa(pub_key):\n        n = pub_key[1]\n        for p in range(2, int(math.sqrt(n))):\n            if n % p == 0:\n                return (p, n//p)",
        security_context="OOP: Static method in CryptoEngine class - factorization attack"
    )
    start_time = time.time()
    private_key = CryptoEngine.crack_rsa(sim_state.wallets['Alice'].pub_key)
    elapsed = (time.time() - start_time) * 1000
    
    if private_key:
        sim_state.execution_tracker.add_step(
            "Factorize RSA Modulus", 
            "passed", 
            f"Found p=61, q=53 in {elapsed:.2f}ms",
            code_snippet="p, q = 61, 53\nphi_n = (p-1) * (q-1)  # 3120\n# Factorization successful",
            security_context="Cryptography: Weak RSA broken - private key components recovered"
        )
        sim_state.execution_tracker.add_step(
            "Calculate Private Key", 
            "checking", 
            "Computing d from phi(n)",
            code_snippet="d = pow(e, -1, phi_n)  # Modular inverse\nprivate_key = (d, n)",
            security_context="Cryptography: Private key d enables transaction signing"
        )
        sim_state.alice_cracked = True
        sim_state.wallets['Alice'].is_compromised = True
        d, n = private_key
        sim_state.execution_tracker.add_step(
            "Calculate Private Key", 
            "passed", 
            f"d={d}",
            code_snippet="wallet.is_compromised = True\n# Eve can now sign as Alice",
            security_context="OOP: Wallet.is_compromised property set - identity theft"
        )
        sim_state.execution_tracker.add_step(
            "Result: SUCCESS", 
            "passed", 
            "Key cracked successfully",
            code_snippet="return private_key  # Attack successful",
            security_context="Security: Wallet compromised - unauthorized access gained"
        )
        sim_state.logs.append(f"[EVE] SUCCESS! Factored n={sim_state.wallets['Alice'].pub_key[1]} into p=61, q=53")
        sim_state.logs.append(f"[EVE] Calculated private key d={d} in {elapsed:.2f}ms")
        sim_state.logs.append("[EVE] Alice's wallet is now COMPROMISED")
        sim_state.execution_tracker.pop_function()
        return jsonify({'success': True, 'message': f'Key cracked in {elapsed:.2f}ms', 'private_key': d})
    
    sim_state.execution_tracker.add_step("Factorize RSA Modulus", "failed", "Factorization failed")
    sim_state.execution_tracker.add_step("Result: FAILED", "failed", "Could not factor modulus")
    sim_state.execution_tracker.pop_function()
    return jsonify({'success': False, 'message': 'Factorization failed'})

@app.route('/api/acquire_hash_power', methods=['POST'])
def acquire_hash_power():
    """Acquire 51% hash power for attack"""
    sim_state.execution_tracker.clear()
    sim_state.execution_tracker.push_function("acquire_hash_power")
    sim_state.execution_tracker.add_step(
        "Start Hash Power Acquisition", 
        "checking", 
        "Renting/buying mining equipment",
        code_snippet="hash_power_distribution = {\n    'Eve': 0.0,\n    'Honest': 100.0\n}",
        security_context="Attack: Acquiring majority hash power"
    )
    sim_state.hash_power_distribution["Eve"] = 51.0
    sim_state.hash_power_distribution["Honest"] = 49.0
    sim_state.execution_tracker.add_step(
        "Acquire Equipment", 
        "passed", 
        "Equipment acquired",
        code_snippet="# Renting/buying mining hardware\n# Acquiring computational resources",
        security_context="Attack: Infrastructure setup"
    )
    sim_state.execution_tracker.add_step(
        "Update Hash Power", 
        "checking", 
        "Eve: 0% → 51%",
        code_snippet="hash_power['Eve'] = 51.0\nhash_power['Honest'] = 49.0",
        security_context="Consensus: Hash power distribution changed"
    )
    sim_state.execution_tracker.add_step(
        "Update Hash Power", 
        "passed", 
        "Eve: 51%, Honest: 49%",
        code_snippet="# Majority hash power acquired\n# Can now mine faster",
        security_context="Attack: 51% attack capability achieved"
    )
    sim_state.execution_tracker.add_step(
        "Result: SUCCESS", 
        "passed", 
        "Now controlling 51% of network",
        code_snippet="return True  # Hash power acquired",
        security_context="Security: Network majority compromised"
    )
    sim_state.logs.append("[EVE] === ACQUIRING 51% HASH POWER ===")
    sim_state.logs.append("[EVE] Renting/buying mining equipment...")
    sim_state.logs.append("[EVE] SUCCESS: Now controlling 51% of network hash power")
    sim_state.logs.append("[EVE] Can mine blocks faster than honest miners")
    sim_state.execution_tracker.pop_function()
    return jsonify({'success': True, 'message': 'Acquired 51% hash power'})

@app.route('/api/mine_attack_block', methods=['POST'])
def mine_attack_block():
    """Mine a block on the attack chain"""
    sim_state.execution_tracker.clear()
    sim_state.execution_tracker.push_function("mine_attack_block")
    sim_state.execution_tracker.add_step(
        "Start Mining Block", 
        "checking", 
        "Beginning block mining process",
        code_snippet="def mine_attack_block():\n    block = Block(index, prev_hash, miner, txs)",
        security_context="Database Layer: Creating block in attack chain"
    )
    
    sim_state.execution_tracker.add_step(
        "Check Hash Power", 
        "checking", 
        f"Eve: {sim_state.hash_power_distribution['Eve']}%",
        code_snippet="if hash_power['Eve'] < 51.0:\n    return 'Insufficient power'",
        security_context="Consensus: 51% hash power enables chain reorganization"
    )
    if sim_state.hash_power_distribution["Eve"] < 51.0:
        sim_state.execution_tracker.add_step("Check Hash Power", "failed", "Insufficient hash power (< 51%)")
        sim_state.execution_tracker.add_step("Result: FAILED", "failed", "Must acquire 51% hash power first")
        sim_state.execution_tracker.pop_function()
        return jsonify({'success': False, 'message': 'Must acquire 51% hash power first'})
    sim_state.execution_tracker.add_step(
        "Check Hash Power", 
        "passed", 
        "Hash power sufficient (>= 51%)",
        code_snippet="hash_power['Eve'] = 51.0  # Majority control",
        security_context="Attack: Majority hash power enables faster mining"
    )
    
    sim_state.execution_tracker.add_step(
        "Check Key Status", 
        "checking", 
        f"Alice cracked: {sim_state.alice_cracked}, Mode: {sim_state.defense_mode}",
        code_snippet="if not alice_cracked:\n    return 'Key not compromised'",
        security_context="Identity Layer: Requires compromised key for theft"
    )
    if not sim_state.alice_cracked and sim_state.defense_mode != "STAKE_CBL":
        sim_state.execution_tracker.add_step("Check Key Status", "failed", "Alice's key not cracked")
        sim_state.execution_tracker.add_step("Result: FAILED", "failed", "Must crack Alice's key first")
        sim_state.execution_tracker.pop_function()
        return jsonify({'success': False, 'message': 'Must crack Alice\'s key first'})
    sim_state.execution_tracker.add_step("Check Key Status", "passed", "Key status OK")
    
    # Initialize attack chain if needed
    sim_state.execution_tracker.add_step(
        "Initialize Attack Chain", 
        "checking", 
        f"Current length: {len(sim_state.attack_chain)}",
        code_snippet="if not attack_chain:\n    attack_chain = [honest_chain.chain[0]]",
        security_context="Database Layer: Forking blockchain from genesis"
    )
    if not sim_state.attack_chain:
        sim_state.attack_chain = [sim_state.honest_chain.chain[0]]
        sim_state.execution_tracker.add_step(
            "Initialize Attack Chain", 
            "passed", 
            "Created fork from genesis",
                code_snippet="class Blockchain:\n    def __init__(self, name):\n        self.chain = [self.create_genesis()]\n    def add_block(self, miner, transactions):\n        block = Block(...)\n        self.chain.append(block)",
                security_context="OOP: Blockchain class (Database Layer) - manages chain of blocks"
        )
        sim_state.logs.append("[EVE] === STARTING ATTACK CHAIN ===")
        sim_state.logs.append("[EVE] Creating secret fork from genesis block...")
        
        # If CBL is enabled and this is a new attack chain, check if previous attack was blocked
        # If so, enable Sybil for Scenario 3
        if sim_state.defense_mode == "CBL":
            sim_state.logs.append("[EVE] Note: In CBL mode, regular attack will be blocked")
            sim_state.logs.append("[EVE] To bypass CBL, use Sybil attack (alternating miners)")
    else:
        sim_state.execution_tracker.add_step("Initialize Attack Chain", "passed", "Using existing attack chain")
    
    # Create theft transaction (only on first block after genesis)
    sim_state.execution_tracker.add_step(
        "Create Transaction", 
        "checking", 
        f"Block number: {len(sim_state.attack_chain)}",
                code_snippet="class Transaction:\n    def __init__(self, tx_id, from_addr, to_addr, amount, signature):\n        self.tx_id = tx_id\n        self.from_addr = from_addr\n        self.to_addr = to_addr\n        self.amount = amount\n        self.signature = signature",
                security_context="OOP: Transaction class (Database Layer) - immutable transaction data"
    )
    if len(sim_state.attack_chain) == 1:  # Only genesis exists, this will be block 1
        if sim_state.defense_mode == "STAKE_CBL":
            # In final defense, Eve tries to double-spend her own funds
            theft_tx = Transaction("tx_attack_eve", "Eve", "Eve", 10.0, "eve_sig_1")
            sim_state.execution_tracker.add_step(
                "Create Transaction", 
                "passed", 
                "Eve -> Eve 10 BTC (double spend)",
                code_snippet="tx = Transaction('tx_attack', 'Eve', 'Eve', 10.0, sig)",
                security_context="Attack: Double-spend attempt"
            )
            sim_state.logs.append("[EVE] Creating transaction: Eve -> Eve 10 BTC (double spend attempt)")
        else:
            theft_tx = Transaction("tx_attack_alice", "Alice", "Eve", 100.0, "stolen_sig_alice")
            sim_state.execution_tracker.add_step(
                "Create Transaction", 
                "passed", 
                "Alice -> Eve 100 BTC (fraudulent)",
                code_snippet="tx = Transaction('tx_attack', 'Alice', 'Eve', 100.0, stolen_sig)",
                security_context="Attack: Fraudulent transaction using stolen key"
            )
            sim_state.logs.append("[EVE] Creating fraudulent transaction: Alice -> Eve 100 BTC")
            sim_state.logs.append("[EVE] Using stolen private key to sign transaction")
        transactions = [theft_tx]
    else:
        sim_state.execution_tracker.add_step("Create Transaction", "passed", "Empty block (no transaction)")
        transactions = []
    
    # Mine block
    prev_hash = sim_state.attack_chain[-1].hash
    miner_name = "Eve"
    
    # For Sybil attack (Scenario 3), alternate miners
    sim_state.execution_tracker.add_step(
        "Determine Miner", 
        "checking", 
        f"Mode: {sim_state.defense_mode}, Sybil: {sim_state.use_sybil}",
        code_snippet="if use_sybil:\n    miner = 'Eve_A' if block_num % 2 else 'Eve_B'",
        security_context="Attack: Sybil attack bypasses CBL by alternating identities"
    )
    if sim_state.defense_mode == "CBL" and sim_state.use_sybil:
        # Sybil attack: alternate between Eve_A and Eve_B
        block_num = len(sim_state.attack_chain)  # Current block number (including genesis)
        if block_num % 2 == 0:  # Even block numbers (2, 4, 6...) -> Eve_B
            miner_name = "Eve_B"
        else:  # Odd block numbers (1, 3, 5...) -> Eve_A
            miner_name = "Eve_A"
        sim_state.execution_tracker.add_step(
            "Determine Miner", 
            "passed", 
            f"Sybil attack: {miner_name}",
            code_snippet="miner_name = 'Eve_A'  # Alternating identity",
            security_context="Attack: Multiple identities bypass static CBL"
        )
    elif sim_state.defense_mode == "STAKE_CBL":
        # In Stake-CBL mode, use Sybil by default (as per Scenario 4)
        block_num = len(sim_state.attack_chain)
        if block_num % 2 == 0:
            miner_name = "Eve_B"
        else:
            miner_name = "Eve_A"
        sim_state.execution_tracker.add_step("Determine Miner", "passed", f"Sybil attack: {miner_name}")
    else:
        sim_state.execution_tracker.add_step("Determine Miner", "passed", f"Regular miner: {miner_name}")
    
    sim_state.execution_tracker.add_step(
        "Mine Block", 
        "checking", 
        f"Miner: {miner_name}, Transactions: {len(transactions)}",
        code_snippet="class Block:\n    def __init__(self, index, prev_hash, miner, transactions):\n        self.index = index\n        self.prev_hash = prev_hash\n        self.miner = miner\n        self.transactions = transactions\n        self.hash = self.compute_hash()",
        security_context="OOP: Block class (Database Layer) - immutable block structure"
    )
    new_block = Block(len(sim_state.attack_chain), prev_hash, miner_name, transactions)
    sim_state.attack_chain.append(new_block)
    sim_state.execution_tracker.add_step(
        "Mine Block", 
        "passed", 
        f"Block {new_block.index} mined successfully",
        code_snippet="attack_chain.append(block)  # Block added",
        security_context="Database Layer: Block appended to chain"
    )
    sim_state.execution_tracker.add_step(
        "Result: SUCCESS", 
        "passed", 
        f"Block {new_block.index} added to attack chain",
        code_snippet="return block  # Mining successful",
        security_context="Attack: Block ready for broadcast"
    )
    
    tx_info = f"{transactions[0].from_addr}->{transactions[0].to_addr} {transactions[0].amount} BTC" if transactions else "Empty"
    sim_state.logs.append(f"[EVE] Mined block {new_block.index} as {miner_name}: {tx_info}")
    sim_state.logs.append(f"[EVE] Attack chain length: {len(sim_state.attack_chain)} (Honest: {len(sim_state.honest_chain.chain)})")
    
    sim_state.execution_tracker.pop_function()
    return jsonify({'success': True, 'message': f'Mined block {new_block.index}', 'block': new_block.to_dict()})

@app.route('/api/broadcast_chain', methods=['POST'])
def broadcast_chain():
    """Broadcast attack chain and resolve fork"""
    sim_state.execution_tracker.clear()
    sim_state.execution_tracker.push_function("broadcast_chain")
    sim_state.execution_tracker.add_step(
        "Start Broadcast", 
        "checking", 
        "Beginning chain broadcast",
        code_snippet="def broadcast_chain():\n    is_valid, reason = validate_fork(attack_chain)",
        security_context="Consensus: Chain validation process"
    )
    
    sim_state.execution_tracker.add_step(
        "Validate Attack Chain", 
        "checking", 
        f"Attack chain length: {len(sim_state.attack_chain)}",
        code_snippet="if len(attack_chain) <= 1:\n    return False  # No blocks",
        security_context="Database Layer: Chain validation"
    )
    if not sim_state.attack_chain or len(sim_state.attack_chain) <= 1:
        sim_state.execution_tracker.add_step("Validate Attack Chain", "failed", "No attack chain to broadcast")
        sim_state.execution_tracker.add_step("Result: FAILED", "failed", "Must mine at least one block")
        sim_state.execution_tracker.pop_function()
        return jsonify({'success': False, 'message': 'No attack chain to broadcast'})
    sim_state.execution_tracker.add_step(
        "Validate Attack Chain", 
        "passed", 
        f"Attack chain has {len(sim_state.attack_chain)} blocks",
        code_snippet="attack_chain_length = len(attack_chain)\n# Chain ready for validation",
        security_context="Database Layer: Chain structure valid"
    )
    
    sim_state.logs.append("[NETWORK] === BROADCASTING ATTACK CHAIN ===")
    sim_state.logs.append(f"[NETWORK] Attack chain length: {len(sim_state.attack_chain)}")
    sim_state.logs.append(f"[NETWORK] Honest chain length: {len(sim_state.honest_chain.chain)}")
    sim_state.logs.append(f"[NETWORK] Defense mode: {sim_state.defense_mode}")
    
    # Show miners in attack chain for debugging
    if len(sim_state.attack_chain) > 1:
        miners = [block.miner for block in sim_state.attack_chain[1:]]
        sim_state.logs.append(f"[NETWORK] Attack chain miners: {', '.join(miners)}")
    
    # Show stake weights if in STAKE_CBL mode
    if sim_state.defense_mode == "STAKE_CBL":
        honest_weight = ConsensusEngine.check_stake_weight(sim_state.honest_chain.chain, sim_state.wallets)
        attack_weight = ConsensusEngine.check_stake_weight(sim_state.attack_chain, sim_state.wallets)
        honest_backing = sim_state.wallets['Alice'].stake + sim_state.wallets['Bob'].stake
        sim_state.logs.append(f"[NETWORK] Honest chain weight: {honest_weight} + stakeholder backing: {honest_backing} = {honest_weight + honest_backing}")
        sim_state.logs.append(f"[NETWORK] Attack chain weight: {attack_weight}")
    
    sim_state.execution_tracker.add_step(
        "Call Consensus Engine", 
        "checking", 
        "Validating fork with consensus rules",
        code_snippet="is_valid, reason = ConsensusEngine.validate_fork(\n    attack_chain, honest_chain, defense_mode)",
        security_context="Consensus: Fork validation using consensus rules"
    )
    is_valid, reason, slashed = sim_state.resolve_fork(sim_state.attack_chain)
    sim_state.execution_tracker.add_step(
        "Call Consensus Engine", 
        "passed" if is_valid else "failed", 
        reason,
        code_snippet="return (is_valid, reason, slashed_miners)",
        security_context="Consensus: Validation result"
    )
    
    if is_valid:
        sim_state.execution_tracker.add_step(
            "Recalculate Balances", 
            "checking", 
            "Processing transactions from new chain",
            code_snippet="class SimulationState:\n    def resolve_fork(self, attack_chain):\n        for block in chain:\n            for tx in block.transactions:\n                wallets[tx.from_addr].balance -= tx.amount\n                wallets[tx.to_addr].balance += tx.amount",
            security_context="OOP: SimulationState.resolve_fork() - balance recalculation method"
        )
        sim_state.execution_tracker.add_step(
            "Recalculate Balances", 
            "passed", 
            f"Alice: {sim_state.wallets['Alice'].balance}, Eve: {sim_state.wallets['Eve'].balance}",
            code_snippet="# Balances updated from accepted chain",
            security_context="Security: Funds restored from valid chain"
        )
        sim_state.execution_tracker.add_step(
            "Result: ACCEPTED", 
            "passed", 
            "Chain reorganization successful",
            code_snippet="honest_chain = attack_chain  # Chain reorganization",
            security_context="Attack: Chain reorganization successful"
        )
        sim_state.logs.append(f"[NETWORK] ✓ CHAIN ACCEPTED: {reason}")
        sim_state.logs.append("[NETWORK] Chain reorganization occurred")
        sim_state.logs.append(f"[NETWORK] Alice balance: {sim_state.wallets['Alice'].balance} BTC")
        sim_state.logs.append(f"[NETWORK] Eve balance: {sim_state.wallets['Eve'].balance} BTC")
        if sim_state.wallets['Alice'].balance == 0:
            sim_state.logs.append("[EVE] ✓ ATTACK SUCCESSFUL - Double spend executed!")
        sim_state.execution_tracker.pop_function()
        return jsonify({'success': True, 'message': reason, 'reorganized': True})
    else:
        sim_state.execution_tracker.add_step("Result: REJECTED", "failed", reason)
        sim_state.logs.append(f"[NETWORK] ✗ CHAIN REJECTED: {reason}")
        if slashed:
            sim_state.logs.append(f"[NETWORK] Stake slashed for: {', '.join(slashed)}")
            sim_state.logs.append(f"[NETWORK] Eve's stake reduced to: {sim_state.wallets['Eve'].stake}")
        sim_state.logs.append("[NETWORK] Honest chain remains valid")
        sim_state.logs.append(f"[NETWORK] Alice balance: {sim_state.wallets['Alice'].balance} BTC (SAFE)")
        sim_state.execution_tracker.pop_function()
        return jsonify({'success': False, 'message': reason, 'slashed': slashed})

@app.route('/api/enable_cbl', methods=['POST'])
def enable_cbl():
    """Enable Static CBL defense"""
    sim_state.execution_tracker.clear()
    sim_state.execution_tracker.push_function("enable_cbl")
    sim_state.execution_tracker.add_step(
        "Start CBL Activation", 
        "checking", 
        "Enabling Static CBL defense",
        code_snippet="def enable_cbl():\n    defense_mode = 'CBL'",
        security_context="Defense: Activating CBL protection"
    )
    sim_state.defense_mode = "CBL"
    sim_state.use_sybil = False  # Reset Sybil flag - Scenario 2 uses regular attack first
    sim_state.execution_tracker.add_step(
        "Set Defense Mode", 
        "passed", 
        "Mode: CBL",
        code_snippet="defense_mode = 'CBL'  # Static CBL enabled",
        security_context="Defense: CBL mode activated"
    )
    sim_state.execution_tracker.add_step(
        "Reset Sybil Flag", 
        "passed", 
        "Sybil: False (Scenario 2)",
        code_snippet="use_sybil = False  # Regular attack first",
        security_context="Security: Sybil attack disabled"
    )
    sim_state.execution_tracker.add_step(
        "Result: SUCCESS", 
        "passed", 
        "CBL defense activated",
        code_snippet="return True  # CBL defense active",
        security_context="Defense: Network protected by CBL"
    )
    sim_state.logs.append("[NETWORK] === STATIC CBL DEFENSE ACTIVATED ===")
    sim_state.logs.append("[NETWORK] Rule: No miner can mine more than 2 consecutive blocks")
    sim_state.logs.append("[NETWORK] This will catch 51% attacks")
    sim_state.logs.append("[NETWORK] Scenario 2: Try regular attack (will be blocked)")
    sim_state.execution_tracker.pop_function()
    return jsonify({'success': True, 'message': 'CBL enabled'})

@app.route('/api/enable_stake_cbl', methods=['POST'])
def enable_stake_cbl():
    """Enable Stake-CBL + ECC defense"""
    sim_state.execution_tracker.clear()
    sim_state.execution_tracker.push_function("enable_stake_cbl")
    sim_state.execution_tracker.add_step(
        "Start Defense Upgrade", 
        "checking", 
        "Enabling Stake-CBL + ECC",
        code_snippet="def enable_stake_cbl():\n    defense_mode = 'STAKE_CBL'",
        security_context="Defense: Activating full protection"
    )
    sim_state.defense_mode = "STAKE_CBL"
    sim_state.execution_tracker.add_step(
        "Set Defense Mode", 
        "passed", 
        "Mode: STAKE_CBL",
        code_snippet="defense_mode = 'STAKE_CBL'  # Full defense",
        security_context="Defense: Stake-CBL mode activated"
    )
    sim_state.execution_tracker.add_step(
        "Upgrade to ECC", 
        "checking", 
        "Upgrading all wallets",
        code_snippet="wallet = Wallet(name, balance, stake, is_ecc=True)\n# ECC keys generated",
        security_context="Cryptography: ECC upgrade prevents key cracking"
    )
    sim_state.wallets['Alice'] = Wallet("Alice", 100.0, 5000, is_ecc=True)
    sim_state.wallets['Bob'] = Wallet("Bob", 50.0, 5000, is_ecc=True)
    sim_state.wallets['Eve'] = Wallet("Eve", 10.0, 200, is_ecc=True)
    sim_state.alice_cracked = False
    sim_state.execution_tracker.add_step(
        "Upgrade to ECC", 
        "passed", 
        "All wallets upgraded",
        code_snippet="# All wallets now use ECC\n# Private keys secure",
        security_context="Security: ECC provides quantum-resistant encryption"
    )
    sim_state.execution_tracker.add_step(
        "Enable Stake-CBL", 
        "passed", 
        "Stake-weighted validation active",
        code_snippet="def check_stake_weight(chain, wallets):\n    return sum(wallets[m].stake for m in miners)",
        security_context="Consensus: Stake-weighted validation enabled"
    )
    sim_state.execution_tracker.add_step(
        "Enable Slashing", 
        "passed", 
        "Attackers will lose stake",
        code_snippet="if attack_failed:\n    wallet.stake -= penalty  # Slashing",
        security_context="Defense: Slashing mechanism active"
    )
    sim_state.execution_tracker.add_step(
        "Result: SUCCESS", 
        "passed", 
        "Full defense activated",
        code_snippet="return True  # Full defense active",
        security_context="Security: Network fully protected"
    )
    sim_state.logs.append("[NETWORK] === STAKE-CBL + ECC DEFENSE ACTIVATED ===")
    sim_state.logs.append("[NETWORK] All wallets upgraded to ECC cryptography")
    sim_state.logs.append("[NETWORK] Stake-weighted CBL enabled")
    sim_state.logs.append("[NETWORK] Attackers will lose stake when caught (slashing)")
    sim_state.logs.append("[NETWORK] Network is now SECURE")
    sim_state.execution_tracker.pop_function()
    return jsonify({'success': True, 'message': 'Stake-CBL + ECC enabled'})

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset simulation to initial state"""
    sim_state.execution_tracker.clear()
    sim_state.execution_tracker.push_function("reset")
    sim_state.execution_tracker.add_step(
        "Reset Simulation", 
        "checking", 
        "Resetting to initial state",
        code_snippet="def reset():\n    wallets = {}\n    chains = []\n    defense_mode = 'LEGACY'",
        security_context="System: State reset"
    )
    sim_state.reset()
    sim_state.execution_tracker.add_step(
        "Reset Simulation", 
        "passed", 
        "All state cleared",
        code_snippet="# All state reset to initial values\n# Ready for new simulation",
        security_context="System: Clean state restored"
    )
    sim_state.execution_tracker.add_step(
        "Result: SUCCESS", 
        "passed", 
        "Simulation reset complete",
        code_snippet="return True  # Reset complete",
        security_context="System: Reset successful"
    )
    sim_state.logs.append("[SYSTEM] Simulation reset to initial state")
    sim_state.execution_tracker.pop_function()
    return jsonify({'success': True, 'message': 'Simulation reset'})

if __name__ == '__main__':
    print("=" * 60)
    print("51% Attack Simulation - Blockchain Security Demo")
    print("=" * 60)
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
