# Step-by-Step Attack Walkthrough

This guide walks through a complete 51% attack scenario from start to finish, showing exactly how to trigger all 5 proof events.

## Scenario 1: Successful 51% Attack (LEGACY Mode)

### Prerequisites
- Flask server running: `python app.py`
- Browser open to: `http://localhost:5000`
- Simulator loaded with initial state

### Step 1: Acquire 51% Hash Power
```
BUTTON: "Acquire 51% Hash Power"
↓
LOG OUTPUT:
  [EVE] === ACQUIRING 51% HASH POWER ===
  [EVE] Renting/buying mining equipment...
  [EVE] SUCCESS: Now controlling 51% of network hash power
  [EVE] Can mine blocks faster than honest miners

STATE AFTER:
  - hash_power: { Eve: 51%, Honest: 49% }
  - Alice cracked: false (not yet)
  - Execution steps logged with code snippets
```

### Step 2: Crack RSA Key
```
BUTTON: "Crack RSA Key"
↓
LOG OUTPUT:
  [EVE] === CRACKING RSA KEY ===
  [EVE] Alice's Public Key: e=17, n=3233
  [EVE] Running factorization algorithm...
  [EVE] SUCCESS! Factored n=3233 into p=61, q=53
  [EVE] Calculated private key d=2753 in X.XXms
  [EVE] Alice's wallet is now COMPROMISED

STATE AFTER:
  - Alice cracked: true
  - Alice compromised: true
  - Eve can now forge Alice's signatures
```

### Step 3: Initiate Secret Fork (Mine First Attack Block)
```
BUTTON: "Mine Attack Block" (FIRST TIME)
↓
LOG OUTPUT:
  [EVE] === STARTING ATTACK CHAIN ===
  [EVE] Creating secret fork from genesis block...
  [EVE] Mined block 1 as Eve: Alice->Eve 100 BTC (fraudulent)
  [EVE] Using stolen private key to sign transaction
  [EVE] Attack chain length: 2 (Honest: 2)

STATE AFTER:
  - attack_chain: [Genesis, Block1(T2: Alice→Eve 100 BTC)]
  - honest_chain: [Genesis, Block1(T1: Alice→Bob 10 BTC)]
  - Both chains have same length, but different transactions
```

**Important**: Note that the attack chain was forked from genesis. Both chains now diverge:
- **Honest chain** contains T1 (the legitimate payment)
- **Attack chain** contains T2 (the fraudulent payment)

### Step 4: Mine More Attack Blocks (Extend Private Chain)
```
BUTTON: "Mine Attack Block" (CLICK 3+ MORE TIMES)
↓
CLICK 1:
  LOG OUTPUT:
    [EVE] Mined block 2 as Eve: Empty
    [EVE] Attack chain length: 3 (Honest: 2)

CLICK 2:
  LOG OUTPUT:
    [EVE] Mined block 3 as Eve: Empty
    [EVE] Attack chain length: 4 (Honest: 2)

CLICK 3:
  LOG OUTPUT:
    [EVE] Mined block 4 as Eve: Empty
    [EVE] Attack chain length: 5 (Honest: 2)

STATE AFTER 3 CLICKS:
  - attack_chain: [Genesis, B1(T2), B2, B3, B4] (length 5)
  - honest_chain: [Genesis, B1(T1)] (length 2)
  - Attack chain is 3 blocks longer

PROOF EVENT 2 TRIGGERED:
  ✓ [PROOF] Event 2: Private Chain Lead Achieved (Attack: 5, Honest: 2)
  - Attack chain has N+2 (minimum) lead required
  - Ready for broadcast
```

### Step 5: Mine Honest Blocks (Extend Public Chain)
```
BUTTON: "Mine Honest Block" (OPTIONAL - Click a few times)
↓
CLICK 1:
  LOG OUTPUT:
    [NETWORK] Miner2 mined block 2 on honest chain
    [NETWORK] Honest chain height: 3

CLICK 2:
  LOG OUTPUT:
    [NETWORK] Miner3 mined block 3 on honest chain
    [NETWORK] Honest chain height: 4

STATE AFTER 2 CLICKS:
  - honest_chain: [Genesis, B1(T1), B2, B3] (length 4)
  - attack_chain: [Genesis, B1(T2), B2, B3, B4] (length 5)
  - Attack still ahead by 1 block

PROOF EVENT 1 TRIGGERED (when honest chain ≥ 7 blocks):
  ⏳ Not yet triggered - need more blocks for 6 confirmations
```

### Step 6: Ensure Event 1 is Triggered (Optional)
If you want to see Event 1 before broadcast:

```
BUTTON: "Mine Honest Block" (3 MORE TIMES until ≥ 7 blocks)
↓
After 3 more clicks:
  - honest_chain: [Genesis, B1(T1), B2, B3, B4, B5, B6, B7] (length 8)
  
PROOF EVENT 1 TRIGGERED:
  ✓ [PROOF] Event 1: Initial Spend Confirmed (T1 has 7 confirmations)
  - T1 in honest chain is now deeply confirmed (6+ blocks)
  - Victim merchant believes payment is safe
  - This is the moment when merchant would hand over goods
```

### Step 7: Broadcast Attack Chain to Network
```
BUTTON: "Broadcast Chain"
↓
VALIDATION PROCESS:
  [NETWORK] === BROADCASTING ATTACK CHAIN ===
  [NETWORK] Attack chain length: 5
  [NETWORK] Honest chain length: 8 (or 4 if you skipped step 5)
  [NETWORK] Defense mode: LEGACY
  
  (Internal validation):
  1. ✓ PoW Validation: Hash chain is valid
  2. ✓ Transaction Validation: All signatures valid
  3. ✓ Defense Mode: LEGACY: Using longest chain rule
  
IF ATTACK CHAIN ≤ HONEST CHAIN:
  [NETWORK] ✗ CHAIN REJECTED: Attack chain too short
  [NETWORK] Honest chain remains valid
  [NETWORK] Alice balance: 100 BTC (SAFE)
  → ATTACK FAILS (return to step 5)

IF ATTACK CHAIN > HONEST CHAIN:
  [NETWORK] ✓ CHAIN ACCEPTED: Longest chain accepted (CBL passed)
  [NETWORK] Chain reorganization occurred
  [NETWORK] Alice balance: 0 BTC
  [NETWORK] Eve balance: 120 BTC
  → ATTACK SUCCEEDS (continue to step 8)
```

**Key observation**: The attack only succeeds if the attack chain is LONGER than the honest chain at the time of broadcast.

### Step 8: Network Reorganization Occurs (Attack Accepted)
```
PROOF EVENT 4 TRIGGERED:
  ✓ [PROOF] Event 4: Network Re-organization Occurred

PROOF EVENT 3 TRIGGERED:
  ✓ [PROOF] Event 3: Conflicting TX Included (T2 present, T1 absent)
  - Detected when attack chain broadcast because T2 in attack chain and T1 not in it

PROOF EVENT 5 TRIGGERED:
  ✓ [PROOF] Event 5: Final Transaction Reversal (T1 reversed, T2 confirmed)

ALL 5 PROOF EVENTS NOW COMPLETE:
  ✓ Event 1: Initial Spend Confirmed
  ✓ Event 2: Private Chain Lead
  ✓ Event 3: Conflicting TX Included
  ✓ Event 4: Network Re-organization
  ✓ Event 5: Final Transaction Reversal

FINAL RESULT:
  [EVE] ✓ ATTACK SUCCESSFUL - All 5 proof events confirmed!

BALANCES:
  Alice: 0 BTC (lost 100 to Eve)
  Bob: 10 BTC (received from original payment, but is still unconfirmed now)
  Eve: 120 BTC (started with 10, received 100 from Alice via double-spend)

CANONICAL CHAIN:
  [Genesis, Block1(T2: Alice→Eve 100 BTC), Block2, Block3, Block4, Block5...]
  
  NOTE: T1 (Alice→Bob) is NO LONGER in canonical chain
        It was in the old honest chain but was reorganized out
        The funds went to Eve (T2) instead
```

---

## Scenario 2: Attack Blocked by CBL Defense

### Setup
```
Same as Scenario 1, but BEFORE cracking key:
BUTTON: "Enable CBL Defense"
↓
[NETWORK] === STATIC CBL DEFENSE ACTIVATED ===
[NETWORK] Rule: No miner can mine more than 2 consecutive blocks
```

### Execution (Steps 1-7 Same)
```
Steps 1-7 same as Scenario 1:
  - Acquire 51% hash power
  - Crack RSA key
  - Mine attack block with T2
  - Mine 3 more attack blocks (Eve mines blocks 1,2,3,4,5)
  - Eve's blocks: 1,2,3,4 are all consecutive from SAME MINER "Eve"
  - This violates CBL rule (max 2 consecutive)
```

### Step 8: Broadcast - CBL Detects Violation
```
BUTTON: "Broadcast Chain"
↓
VALIDATION PROCESS:
  [NETWORK] === BROADCASTING ATTACK CHAIN ===
  [NETWORK] Attack chain length: 5
  [NETWORK] Defense mode: CBL
  
  (Internal validation):
  1. ✓ PoW Validation: Hash chain is valid
  2. ✓ Transaction Validation: All signatures valid
  3. ✗ Defense Mode: CBL - Checking consecutive block limit
  
  ✗ CBL VIOLATION: Eve mined 4 consecutive blocks (limit: 2)
  ✗ Block 1,2,3,4 all have miner="Eve"

RESULT:
  [NETWORK] ✗ CHAIN REJECTED: CBL violation detected
  [NETWORK] Honest chain remains valid
  [NETWORK] Alice balance: 100 BTC (SAFE)

PROOF EVENTS:
  ✓ Event 1: Initial Spend Confirmed (if honest chain ≥ 7)
  ✓ Event 2: Private Chain Lead (before broadcast)
  ✓ Event 3: Conflicting TX Included (T2 in chain)
  ✗ Event 4: NOT TRIGGERED - Attack rejected before re-org
  ✗ Event 5: NOT TRIGGERED - No reversal

ATTACK RESULT: FAILED
  - Attacker's chain rejected
  - Victim's payment T1 remains safe
  - Eve's malicious T2 never confirmed
  - Funds remain with Alice/Bob
```

### Optional: Eve Attempts Sybil Attack
```
[EVE] Attack blocked by CBL! Switching to Sybil attack (Scenario 3)
[EVE] Creating Eve_A and Eve_B identities to bypass CBL...

BUTTON: "Mine Attack Block" (to continue with Sybil)
↓
NEW BEHAVIOR:
  - Attack chain is reset (or continued from where blocked)
  - Eve_A mines block 1 with T2
  - Eve_B mines block 2 (empty)
  - Eve_A mines block 3 (empty)
  - Eve_B mines block 4 (empty)
  
  No single identity mines >2 consecutive blocks!
  CBL check now passes because miner names alternate:
    Block 1: Eve_A
    Block 2: Eve_B
    Block 3: Eve_A
    Block 4: Eve_B
  
RESULT:
  - CBL is bypassed
  - Sybil identities trick the defense
  - Attack can succeed if chain becomes longer
```

---

## Scenario 3: Attack Defeated by Stake-CBL Defense

### Setup
```
BUTTON: "Enable Stake-CBL + ECC"
↓
[NETWORK] === STAKE-CBL + ECC DEFENSE ACTIVATED ===
[NETWORK] All wallets upgraded to ECC cryptography
[NETWORK] Stake-weighted CBL enabled
[NETWORK] Attackers will lose stake when caught (slashing)

WALLET STAKES:
  Alice: 5000 (high stake)
  Bob: 5000 (high stake)
  Eve: 200 (low stake)
```

### Step 2: Crack Key - Fails on ECC
```
BUTTON: "Crack RSA Key"
↓
LOG OUTPUT:
  [EVE] Attempting to crack ECC key...
  [EVE] FAILED: ECC cryptography is computationally secure
  [EVE] Cannot factorize elliptic curve discrete logarithm problem

STATE:
  - Alice cracked: false
  - Alice's key is UNCRACKABLE (ECC is secure)
  - Eve cannot forge Alice's signature
```

### Step 3-7: Mine Attack Chain (Different T2)
```
Since Eve can't crack Alice's key, she uses a different strategy:

BUTTON: "Mine Attack Block"
↓
[EVE] Creating transaction: Eve -> Eve 10 BTC (double spend attempt)
[EVE] Mined block 1 as Eve_A: Eve->Eve 10 BTC
[EVE] Mined block 2 as Eve_B: Empty
[EVE] Mined block 3 as Eve_A: Empty
[EVE] Mined block 4 as Eve_B: Empty

STATE:
  - attack_chain: [Genesis, B1(Eve→Eve 10 BTC), B2, B3, B4]
  - Eve tries to double-spend her own funds in STAKE_CBL mode
  - Attack chain is longer than honest chain
  - Ready for broadcast
```

### Step 8: Broadcast - Stake Weight Check
```
BUTTON: "Broadcast Chain"
↓
VALIDATION PROCESS:
  [NETWORK] === BROADCASTING ATTACK CHAIN ===
  [NETWORK] Calculating economic weight...
  
  Honest chain weight:
    - Miner1 stake: 0 (not a wallet in our system)
    - Alice backing: 5000 (high stake validates honest chain)
    - Bob backing: 5000 (high stake validates honest chain)
    - Total honest weight: 10000
  
  Attack chain weight:
    - Eve_A stake: 100 (half of Eve's stake)
    - Eve_B stake: 100 (half of Eve's stake)
    - Total attack weight: 200
  
  COMPARISON: Attack (200) < Honest (10000)
  
RESULT:
  [NETWORK] ✗ CHAIN REJECTED: Insufficient stake weight
  [NETWORK] Attack={200}, Honest={10000}
  [NETWORK] Stake slashed for: Eve_A, Eve_B
  [NETWORK] Eve's stake reduced to: 75 (from 200)

SLASHING BREAKDOWN:
  - Eve_A penalty: -25 coins
  - Eve_B penalty: -25 coins
  - Total lost: 50 coins (25% of stake)
  - New Eve stake: 200 - 50 = 150

PROOF EVENTS:
  ✓ Event 1: Initial Spend Confirmed (if ≥7 blocks)
  ✓ Event 2: NOT YET - Attack chain not longer enough
  ✓ Event 3: Conflicting TX included (Eve→Eve in chain)
  ✗ Event 4: NOT TRIGGERED - Attack rejected
  ✗ Event 5: NOT TRIGGERED - No reversal

ATTACK RESULT: FAILED + PUNISHED
  - Attack rejected
  - Attacker economically slashed
  - Repeated attacks become increasingly expensive
  - Eventually, Eve's stake → 0 (cannot attack anymore)
```

---

## Quick Reference: How to Trigger Each Proof Event

### Event 1: Initial Spend Confirmation
```
Requirement: T1 confirmed with 6+ confirmations
Action: Click "Mine Honest Block" until honest_chain.length ≥ 7
Verification: Log shows "[PROOF] ✓ Event 1: Initial Spend Confirmed"
```

### Event 2: Private Chain Lead
```
Requirement: Attack chain ≥ 2 blocks longer than honest chain
Action: Mine attack blocks while keeping honest chain short
Verification: Log shows "[PROOF] ✓ Event 2: Private Chain Lead Achieved"
```

### Event 3: Conflicting TX Inclusion
```
Requirement: T2 in attack chain, T1 NOT in attack chain
Condition: Automatically satisfied when first attack block mined (contains T2)
Verification: Log shows "[PROOF] ✓ Event 3: Conflicting TX Included"
Timeline: Triggered when block with T2 is detected
```

### Event 4: Network Re-organization
```
Requirement: Consensus accepts attack chain
Action: Click "Broadcast Chain" when attack_chain > honest_chain
Condition: LEGACY mode (no defense), or defenses bypassed
Verification: Log shows "[PROOF] ✓ Event 4: Network Re-organization Occurred"
```

### Event 5: Final Transaction Reversal
```
Requirement: T1 in old_chain but NOT in new_chain, AND T2 in new_chain
Condition: Event 4 must succeed first
Verification: Log shows "[PROOF] ✓ Event 5: Final Transaction Reversal"
Also verify: Alice balance = 0, Eve balance = 110+
Timeline: Triggered immediately after Event 4
```

---

## Summary

- **Scenario 1 (LEGACY)**: All 5 events can be triggered
- **Scenario 2 (CBL)**: Events 1-3 possible, 4-5 blocked
- **Scenario 3 (Sybil)**: Events 1-3 possible with different identities
- **Scenario 4 (STAKE_CBL)**: Events 1-3 possible but Event 4-5 blocked by stake weight

Each scenario demonstrates different aspects of blockchain security!
