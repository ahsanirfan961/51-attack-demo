#!/usr/bin/env python3
"""Test exact scenario: 3 honest blocks + 4 attack blocks with CBL"""

from app import Block, ConsensusEngine

# Create genesis
genesis = Block(0, "0", "Genesis", [])

# Scenario: 3 honest blocks mined (meaning 3 new blocks + genesis = 4 total)
# Miners: Miner1, Miner2, Miner3 (all different)
honest_blocks = [
    genesis,
    Block(1, genesis.hash, "Miner1", []),
    Block(2, None, "Miner2", []),
    Block(3, None, "Miner3", []),
]
# Fix prev_hash
for i in range(1, len(honest_blocks)):
    honest_blocks[i].prev_hash = honest_blocks[i-1].hash

# Attack: 4 blocks (all by Eve)
attack_blocks = [
    genesis,
    Block(1, genesis.hash, "Eve", []),
    Block(2, None, "Eve", []),
    Block(3, None, "Eve", []),
    Block(4, None, "Eve", []),
]
# Fix prev_hash
for i in range(1, len(attack_blocks)):
    attack_blocks[i].prev_hash = attack_blocks[i-1].hash

print("=== SCENARIO: 3 Honest Blocks + 4 Attack Blocks ===")
print(f"Honest chain: {len(honest_blocks)} blocks, miners: {[b.miner for b in honest_blocks]}")
print(f"Attack chain: {len(attack_blocks)} blocks, miners: {[b.miner for b in attack_blocks]}")
print()

# Test CBL on attack chain
is_valid, reason = ConsensusEngine.check_static_cbl(attack_blocks, max_consecutive=2)
print(f"CBL Check on Attack Chain: {is_valid}")
if not is_valid:
    print(f"✓ CORRECT - Attack rejected: {reason}")
else:
    print(f"✗ ERROR - Attack should be rejected!")
    print(f"  Reason given: {reason}")
