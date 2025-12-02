#!/usr/bin/env python3
"""Test CBL logic with specific scenario"""

from app import Block, ConsensusEngine

# Create a genesis block
genesis = Block(0, "0", "Genesis", [])

# Create attack chain: Genesis + 4 blocks all mined by "Eve"
attack_blocks = [
    genesis,
    Block(1, genesis.hash, "Eve", []),
    Block(2, None, "Eve", []),  # prev_hash will be set
    Block(3, None, "Eve", []),
    Block(4, None, "Eve", []),
]

# Fix prev_hash
for i in range(1, len(attack_blocks)):
    attack_blocks[i].prev_hash = attack_blocks[i-1].hash

print("Attack chain miners:", [b.miner for b in attack_blocks])
print(f"Attack chain length: {len(attack_blocks)}")

# Test CBL check
is_valid, reason = ConsensusEngine.check_static_cbl(attack_blocks, max_consecutive=2)
print(f"\nCBL Check Result: {is_valid}")
print(f"Reason: {reason}")

# Expected: False (Eve mined 4 consecutive blocks, limit is 2)
if not is_valid:
    print("✓ CORRECT: CBL correctly detected violation")
else:
    print("✗ ERROR: CBL should have rejected this chain!")
