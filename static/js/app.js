// Blockchain Visualization using vis.js
let network = null;
let nodes = null;
let edges = null;

// Code Execution Flow Visualization
let showDetails = false;

const API_BASE = '';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeVisualization();
    initializeCodeFlowVisualization();
    setupEventListeners();
    startPolling();
});

function initializeVisualization() {
    const container = document.getElementById('blockchain-visualization');

    const options = {
        nodes: {
            shape: 'box',
            font: {
                size: 12,
                face: 'Arial',
                color: '#ffffff',
                bold: true
            },
            borderWidth: 3,
            shadow: true,
            margin: 15,
            widthConstraint: {
                maximum: 150
            }
        },
        edges: {
            arrows: {
                to: { enabled: true, scaleFactor: 1.5, type: 'arrow' }
            },
            width: 4,
            smooth: {
                type: 'straightCross',
                roundness: 0
            }
        },
        layout: {
            hierarchical: {
                direction: 'LR',
                sortMethod: 'directed',
                levelSeparation: 200,
                nodeSpacing: 250,
                treeSpacing: 300
            }
        },
        physics: {
            enabled: false
        },
        interaction: {
            dragNodes: true,
            zoomView: true,
            dragView: true
        }
    };

    nodes = new vis.DataSet([]);
    edges = new vis.DataSet([]);
    network = new vis.Network(container, { nodes, edges }, options);
}

function initializeCodeFlowVisualization() {
    const container = document.getElementById('code-flow-visualization');
    if (!container) return;

    const options = {
        nodes: {
            shape: 'box',
            font: {
                size: 9,
                face: 'Courier New, monospace',
                color: '#ffffff',
                bold: true,
                multi: true,
                align: 'left'
            },
            borderWidth: 2,
            shadow: true,
            margin: 12,
            widthConstraint: {
                maximum: 300
            },
            heightConstraint: {
                minimum: 80,
                maximum: 200
            },
            shapeProperties: {
                borderRadius: 5
            },
            labelHighlightBold: false
        },
        edges: {
            arrows: {
                to: { enabled: true, scaleFactor: 1.2, type: 'arrow' }
            },
            width: 2,
            smooth: {
                type: 'straightCross',
                roundness: 0
            },
            font: {
                size: 10,
                align: 'middle'
            }
        },
        layout: {
            hierarchical: {
                direction: 'TD',
                sortMethod: 'directed',
                levelSeparation: 120,
                nodeSpacing: 120,
                treeSpacing: 200,
                blockShifting: true,
                edgeMinimization: true
            }
        },
        physics: {
            enabled: false
        },
        interaction: {
            dragNodes: false,
            zoomView: true,
            dragView: true
        }
    };

    // Use simple HTML/CSS instead of vis.js for code execution flow
    container.innerHTML = '<div id="code-execution-chain" class="code-execution-chain"></div>';
}

function setupEventListeners() {
    document.getElementById('btn-crack-rsa').addEventListener('click', crackRSA);
    document.getElementById('btn-acquire-hash').addEventListener('click', acquireHashPower);
    document.getElementById('btn-mine-block').addEventListener('click', mineAttackBlock);
    document.getElementById('btn-mine-honest').addEventListener('click', mineHonestBlock);
    document.getElementById('btn-broadcast').addEventListener('click', broadcastChain);
    document.getElementById('btn-enable-cbl').addEventListener('click', enableCBL);
    document.getElementById('btn-enable-stake').addEventListener('click', enableStakeCBL);
    document.getElementById('btn-reset').addEventListener('click', resetSimulation);
    document.getElementById('btn-expand-details').addEventListener('click', toggleDetails);
    document.getElementById('btn-clear-flow').addEventListener('click', clearFlow);
}

function startPolling() {
    updateState();
    setInterval(updateState, 500);
}

async function updateState() {
    try {
        const response = await fetch(`${API_BASE}/api/state`);
        const state = await response.json();

        updateBlockchainVisualization(state);
        updateWalletBalances(state.wallets);
        updateLogs(state.logs);
        updateStatus(state);
        updateDefenseStatus(state);
        updateCodeFlowVisualization(state);
        updateFunctionStack(state);
    } catch (error) {
        console.error('Error updating state:', error);
    }
}

function updateBlockchainVisualization(state) {
    const nodeMap = new Map();
    const edgeList = [];
    const honestChain = state.honest_chain;
    const attackChain = state.attack_chain;

    const HONEST_Y = 1.0;
    const ATTACK_Y = 0.0;

    // Add honest chain nodes (top)
    honestChain.forEach((block, idx) => {
        const nodeId = `H${block.index}`;
        const isGenesis = block.index === 0;

        const txInfo = block.transactions.length > 0
            ? `${block.transactions[0].from_addr}->${block.transactions[0].to_addr} ${block.transactions[0].amount} BTC`
            : 'Genesis';

        nodeMap.set(nodeId, {
            id: nodeId,
            label: `#${block.index}\n${block.miner.substring(0, 10)}\n${txInfo}`,
            color: isGenesis ? '#ff9800' : '#1976d2',
            borderWidth: isGenesis ? 5 : 3,
            borderColor: '#000000',
            shape: isGenesis ? 'ellipse' : 'box',
            font: {
                size: isGenesis ? 14 : 11,
                bold: true
            },
            x: idx * 300,
            y: HONEST_Y * 200,
            fixed: { x: true, y: true },
            level: idx
        });

        if (idx > 0) {
            edgeList.push({
                from: `H${honestChain[idx - 1].index}`,
                to: nodeId,
                color: '#1976d2',
                width: 4,
                arrows: 'to'
            });
        }
    });

    // Add attack chain nodes (bottom)
    if (attackChain && attackChain.length > 0) {
        attackChain.forEach((block, idx) => {
            if (block.index === 0) return; // Skip genesis

            const nodeId = `A${block.index}_${idx}`;
            const txInfo = block.transactions.length > 0
                ? `${block.transactions[0].from_addr}->${block.transactions[0].to_addr} ${block.transactions[0].amount} BTC`
                : 'Empty';

            nodeMap.set(nodeId, {
                id: nodeId,
                label: `#${block.index}\n${block.miner.substring(0, 10)}\n${txInfo}\n[ATTACK]`,
                color: '#d32f2f',
                borderWidth: 3,
                borderColor: '#000000',
                shape: 'box',
                font: {
                    size: 11,
                    bold: true
                },
                x: idx * 300,
                y: ATTACK_Y * 200,
                fixed: { x: true, y: true },
                level: idx
            });

            const prevId = idx > 1 ? `A${attackChain[idx - 1].index}_${idx - 1}` : 'H0';
            edgeList.push({
                from: prevId,
                to: nodeId,
                color: '#d32f2f',
                width: 4,
                dashes: [5, 5],
                arrows: 'to'
            });
        });
    }

    // Update network
    nodes.clear();
    edges.clear();

    if (nodeMap.size > 0) {
        nodes.add(Array.from(nodeMap.values()));
        edges.add(edgeList);

        setTimeout(() => {
            network.fit({
                animation: {
                    duration: 300,
                    easingFunction: 'easeInOutQuad'
                }
            });
        }, 100);
    }
}

function updateWalletBalances(wallets) {
    const container = document.getElementById('wallet-balances');
    container.innerHTML = '';

    Object.values(wallets).forEach(wallet => {
        const div = document.createElement('div');
        div.className = `wallet-item ${wallet.name.toLowerCase()}`;

        const balanceColor = wallet.balance === 0 ? 'red' : (wallet.balance < wallet.original_balance ? 'orange' : 'green');

        div.innerHTML = `
            <div>
                <div class="wallet-name">${wallet.name}</div>
                <div class="wallet-stake">Stake: ${wallet.stake} | Original: ${wallet.original_stake || wallet.stake}</div>
            </div>
            <div class="wallet-balance" style="color: ${balanceColor}; font-weight: bold;">
                ${wallet.balance} BTC
            </div>
        `;
        container.appendChild(div);
    });
}

function updateLogs(logs) {
    const container = document.getElementById('logs');
    container.innerHTML = '';

    logs.forEach(log => {
        const div = document.createElement('div');
        div.className = 'log-entry';

        if (log.includes('SUCCESS') || log.includes('✓')) {
            div.className += ' success';
        } else if (log.includes('FAILED') || log.includes('BLOCKED') || log.includes('✗') || log.includes('REJECTED')) {
            div.className += ' error';
        } else if (log.includes('NETWORK') || log.includes('=== ')) {
            div.className += ' info';
        } else if (log.includes('EVE')) {
            div.className += ' warning';
        }

        div.textContent = log;
        container.appendChild(div);
    });

    container.scrollTop = container.scrollHeight;
}

function updateStatus(state) {
    document.getElementById('alice-status').textContent = state.alice_cracked ? 'Cracked' : 'Secure';
    document.getElementById('alice-status').className = state.alice_cracked ? 'status-badge cracked' : 'status-badge';

    const evePower = state.hash_power?.Eve || 0;
    document.getElementById('hash-power').textContent = `${evePower}%`;
    document.getElementById('hash-power').className = evePower >= 51 ? 'status-badge cracked' : 'status-badge';

    const attackBlocks = state.attack_chain?.length || 0;
    document.getElementById('attack-blocks').textContent = Math.max(0, attackBlocks - 1); // Exclude genesis
}

function updateDefenseStatus(state) {
    const mode = state.defense_mode || 'LEGACY';
    document.getElementById('defense-mode').textContent = mode;
    document.getElementById('defense-mode').className = mode === 'LEGACY' ? 'defense-badge inactive' :
        mode === 'CBL' ? 'defense-badge active' :
            'defense-badge active';

    const isECC = state.wallets?.Alice?.is_ecc || false;
    document.getElementById('crypto-status').textContent = isECC ? 'ECC (Secure)' : 'RSA (Weak)';
    document.getElementById('crypto-status').className = isECC ? 'defense-badge active' : 'defense-badge inactive';

    document.getElementById('cbl-status').textContent = mode === 'CBL' || mode === 'STAKE_CBL' ? 'ACTIVE' : 'INACTIVE';
    document.getElementById('cbl-status').className = mode === 'CBL' || mode === 'STAKE_CBL' ? 'defense-badge active' : 'defense-badge inactive';

    document.getElementById('stake-status').textContent = mode === 'STAKE_CBL' ? 'ACTIVE' : 'INACTIVE';
    document.getElementById('stake-status').className = mode === 'STAKE_CBL' ? 'defense-badge active' : 'defense-badge inactive';
}

// API Calls
async function crackRSA() {
    try {
        const response = await fetch(`${API_BASE}/api/crack_rsa`, {
            method: 'POST'
        });
        const result = await response.json();
        if (!result.success) {
            alert(result.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error cracking RSA key');
    }
}

async function acquireHashPower() {
    try {
        const response = await fetch(`${API_BASE}/api/acquire_hash_power`, {
            method: 'POST'
        });
        const result = await response.json();
        if (!result.success) {
            alert(result.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error acquiring hash power');
    }
}

async function mineAttackBlock() {
    try {
        const response = await fetch(`${API_BASE}/api/mine_attack_block`, {
            method: 'POST'
        });
        const result = await response.json();
        if (!result.success) {
            alert(result.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error mining block');
    }
}

async function mineHonestBlock() {
    try {
        const response = await fetch(`${API_BASE}/api/mine_honest_block`, {
            method: 'POST'
        });
        const result = await response.json();
        if (!result.success) {
            alert(result.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error mining honest block');
    }
}

async function broadcastChain() {
    try {
        const response = await fetch(`${API_BASE}/api/broadcast_chain`, {
            method: 'POST'
        });
        const result = await response.json();
        if (result.success) {
            alert(`Chain accepted: ${result.message}`);
        } else {
            alert(`Chain rejected: ${result.message}`);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error broadcasting chain');
    }
}

async function enableCBL() {
    try {
        const response = await fetch(`${API_BASE}/api/enable_cbl`, {
            method: 'POST'
        });
        const result = await response.json();
        if (!result.success) {
            alert(result.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error enabling CBL');
    }
}

async function enableStakeCBL() {
    try {
        const response = await fetch(`${API_BASE}/api/enable_stake_cbl`, {
            method: 'POST'
        });
        const result = await response.json();
        if (!result.success) {
            alert(result.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error enabling Stake-CBL');
    }
}

function toggleDetails() {
    showDetails = !showDetails;
    document.getElementById('btn-expand-details').textContent = showDetails ? 'Hide Details' : 'Toggle Details';
    // Re-render with new detail level
    updateState();
}

function clearFlow() {
    const container = document.getElementById('code-execution-chain');
    if (container) {
        container.innerHTML = '<div class="code-exec-empty">Flow cleared. Click a button to see execution flow.</div>';
    }
    if (document.getElementById('function-stack')) {
        document.getElementById('function-stack').innerHTML = '';
    }
}

function updateCodeFlowVisualization(state) {
    const container = document.getElementById('code-execution-chain');
    if (!container || !state.execution_tracker) return;

    const steps = state.execution_tracker.steps || [];
    if (steps.length === 0) {
        container.innerHTML = '<div class="code-exec-empty">No code execution yet. Click a button to see execution flow.</div>';
        return;
    }

    // Build code execution chain
    let html = '<div class="code-chain">';

    steps.forEach((step, idx) => {
        const status = step.status || 'checking';
        const statusClass = status === 'passed' ? 'passed' : (status === 'failed' ? 'failed' : 'checking');
        const functionName = step.function || 'unknown';

        html += `<div class="code-block ${statusClass}" data-index="${idx}">`;
        html += `<div class="code-block-header">`;
        html += `<span class="code-block-number">#${idx + 1}</span>`;
        html += `<span class="code-block-function">${functionName}()</span>`;
        html += `<span class="code-block-step">${step.step}</span>`;
        html += `</div>`;

        // Code snippet
        if (step.code_snippet) {
            html += `<div class="code-snippet">`;
            html += `<pre><code>${escapeHtml(step.code_snippet)}</code></pre>`;
            html += `</div>`;
        }

        // Security context
        if (step.security_context) {
            html += `<div class="code-context">`;
            html += `<span class="context-label">[${step.security_context}]</span>`;
            html += `</div>`;
        }

        // Details if no code snippet
        if (step.details && !step.code_snippet) {
            html += `<div class="code-details">${escapeHtml(step.details)}</div>`;
        }

        html += `</div>`;

        // Arrow connector (except for last step)
        if (idx < steps.length - 1) {
            html += `<div class="code-arrow">→</div>`;
        }
    });

    html += '</div>';
    container.innerHTML = html;

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateFunctionStack(state) {
    const container = document.getElementById('function-stack');
    if (!container || !state.execution_tracker) return;

    const callStack = state.execution_tracker.call_stack || [];
    const steps = state.execution_tracker.steps || [];
    const currentFunction = state.execution_tracker.current_function;

    container.innerHTML = '';

    if (callStack.length === 0 && steps.length === 0) {
        container.innerHTML = '<div style="color: #757575; font-style: italic;">No active execution</div>';
        return;
    }

    // Show call stack
    const stackHeader = document.createElement('div');
    stackHeader.style.fontWeight = 'bold';
    stackHeader.style.marginBottom = '10px';
    stackHeader.textContent = 'Function Call Stack:';
    container.appendChild(stackHeader);

    callStack.forEach((func, idx) => {
        const div = document.createElement('div');
        div.className = 'function-stack-item';
        div.style.marginLeft = `${idx * 20}px`;

        if (func === currentFunction) {
            div.className += ' checking';
        }

        div.textContent = `${idx + 1}. ${func}()`;
        container.appendChild(div);
    });

    // Show current step if available
    if (steps.length > 0) {
        const lastStep = steps[steps.length - 1];
        const currentDiv = document.createElement('div');
        currentDiv.className = `function-stack-item ${lastStep.status}`;
        currentDiv.style.marginTop = '10px';
        currentDiv.style.fontWeight = 'bold';
        currentDiv.textContent = `Current: ${lastStep.step}`;
        if (lastStep.details) {
            const detailsDiv = document.createElement('div');
            detailsDiv.style.fontSize = '0.85em';
            detailsDiv.style.color = '#666';
            detailsDiv.textContent = lastStep.details;
            currentDiv.appendChild(detailsDiv);
        }
        container.appendChild(currentDiv);
    }
}

async function resetSimulation() {
    if (confirm('Reset the simulation to initial state?')) {
        try {
            await fetch(`${API_BASE}/api/reset`, {
                method: 'POST'
            });
            clearFlow();
        } catch (error) {
            console.error('Error:', error);
            alert('Error resetting simulation');
        }
    }
}
