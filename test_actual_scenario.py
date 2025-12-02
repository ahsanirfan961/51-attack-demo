#!/usr/bin/env python3
"""Test scenario: Equal length chains with 3 honest blocks vs 4 attack blocks"""

from app import Block, ConsensusEngine

# Genesis
genesis = Block(0, "0", "Genesis", [])

# Honest: Genesis + 3 blocks (different miners) + initial Miner1 = 5 total
# Wait, initial honest chain has 1 block already (Miner1), so:
# 1 (Miner1) + 3 more = 4 total honest blocks

honest_blocks = [
    genesis,
    Block(1, genesis.hash, "Miner1", []),  # Initial block
    Block(2, None, "Miner2", []),           # Mine Honest 1
    Block(3, None, "Miner3", []),           # Mine Honest 2
    Block(4, None, "Miner1", []),           # Mine Honest 3 (rotates back)
]
for i in range(1, len(honest_blocks)):
    honest_blocks[i].prev_hash = honest_blocks[i-1].hash

# Attack: Genesis + 4 blocks all Eve
attack_blocks = [
    genesis,
    Block(1, genesis.hash, "Eve", []),
    Block(2, None, "Eve", []),
    Block(3, None, "Eve", []),
    Block(4, None, "Eve", []),
]
for i in range(1, len(attack_blocks)):
    attack_blocks[i].prev_hash = attack_blocks[i-1].hash

print("=== ACTUAL SCENARIO: Initial + 3 Honest Blocks vs 4 Attack Blocks ===")
print(f"Honest chain: {len(honest_blocks)} blocks, miners: {[b.miner for b in honest_blocks]}")
print(f"Attack chain: {len(attack_blocks)} blocks, miners: {[b.miner for b in attack_blocks]}")
print()

# Test CBL on attack chain
is_valid, reason = ConsensusEngine.check_static_cbl(attack_blocks, max_consecutive=2)
print(f"CBL Check: {is_valid}")
if not is_valid:
    print(f"✓ CORRECT - {reason}")
else:
    print(f"✗ ERROR - Should reject!")
