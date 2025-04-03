# CSC242: Project 3 - Uncertainty Modeling
# Date: April 3, 2025

import sys
import random
import math
from collections import defaultdict

# Utility function to parse Bayesian network file
def parse_network(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    idx = 0
    num_vars = int(lines[idx])
    idx += 1

    vars = []
    domains = {}
    while len(vars) < num_vars:
        parts = lines[idx].split()
        var = parts[0]
        vals = parts[1:]
        vars.append(var)
        domains[var] = vals
        idx += 1

    num_cpts = int(lines[idx])
    idx += 1

    cpts = {}
    while idx < len(lines):
        header = lines[idx].split()
        idx += 1

        if len(header) == 1:
            child = header[0]
            parents = []
        else:
            child = header[0]
            parents = header[1:]

        table = []
        while idx < len(lines) and not lines[idx].startswith('#') and lines[idx]:
            probs = list(map(float, lines[idx].split()))
            table.append(probs)
            idx += 1
        idx += 1

        cpts[child] = {'parents': parents, 'table': table}

    return vars, domains, cpts

# Compute full joint probability
def joint_prob(assignment, vars, domains, cpts):
    prob = 1.0
    for var in vars:
        val_index = domains[var].index(assignment[var])
        parents = cpts[var]['parents']
        table = cpts[var]['table']

        if not parents:
            row = table[0]
        else:
            parent_vals = [assignment[p] for p in parents]
            parent_idx = 0
            multiplier = 1
            for i, p in enumerate(reversed(parents)):
                multiplier *= len(domains[p])
                parent_idx += domains[p].index(parent_vals[len(parents)-1-i]) * (multiplier // len(domains[p]))
            row = table[parent_idx]

        prob *= row[val_index]
    return prob

# Exact inference

def exact_inference(query, evidence, vars, domains, cpts):
    hidden_vars = [v for v in vars if v != query and v not in evidence]
    counts = [0.0] * len(domains[query])

    def enumerate_all(i, assignment):
        if i == len(hidden_vars):
            for j, val in enumerate(domains[query]):
                assignment[query] = val
                counts[j] += joint_prob(assignment, vars, domains, cpts)
        else:
            var = hidden_vars[i]
            for val in domains[var]:
                assignment[var] = val
                enumerate_all(i+1, assignment)

    enumerate_all(0, dict(evidence))
    total = sum(counts)
    return [round(c/total, 4) for c in counts]

# Rejection sampling
def rejection_sample(query, evidence, vars, domains, cpts, N):
    counts = [0.0] * len(domains[query])
    for _ in range(N):
        sample = {} 
        consistent = True
        for var in vars:
            parents = cpts[var]['parents']
            table = cpts[var]['table']

            if not parents:
                probs = table[0]
            else:
                parent_vals = [sample[p] for p in parents]
                parent_idx = 0
                multiplier = 1
                for i, p in enumerate(reversed(parents)):
                    multiplier *= len(domains[p])
                    parent_idx += domains[p].index(parent_vals[len(parents)-1-i]) * (multiplier // len(domains[p]))
                probs = table[parent_idx]

            r = random.random()
            cum = 0.0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    sample[var] = domains[var][i]
                    break

            if var in evidence and sample[var] != evidence[var]:
                consistent = False
                break

        if consistent:
            val = sample[query]
            counts[domains[query].index(val)] += 1

    total = sum(counts)
    return [round(c/total, 4) if total > 0 else 0.0 for c in counts]

# Gibbs sampling
def gibbs_sample(query, evidence, vars, domains, cpts, N, burn=100):
    non_evidence = [v for v in vars if v not in evidence]
    state = dict(evidence)
    for v in non_evidence:
        state[v] = random.choice(domains[v])

    counts = [0.0] * len(domains[query])
    for i in range(N+burn):
        for v in non_evidence:
            dist = []
            for val in domains[v]:
                state[v] = val
                dist.append(joint_prob(state, vars, domains, cpts))
            total = sum(dist)
            r = random.random()
            cum = 0.0
            for j, p in enumerate(dist):
                cum += p/total
                if r <= cum:
                    state[v] = domains[v][j]
                    break

        if i >= burn:
            counts[domains[query].index(state[query])] += 1

    total = sum(counts)
    return [round(c/total, 4) if total > 0 else 0.0 for c in counts]

# Main REPL loop
def repl():
    vars = []
    domains = {}
    cpts = {}
    print("BayesNet REPL. Use 'help' to see commands.")
    while True:
        try:
            cmd = input("BN > ").strip()
            if not cmd:
                continue
            if cmd == 'quit':
                break
            elif cmd.startswith('load'):
                _, fname = cmd.split()
                vars, domains, cpts = parse_network(fname)
                print("Loaded network with:", vars)
            elif cmd.startswith('xquery') or cmd.startswith('rquery') or cmd.startswith('gquery'):
                parts = cmd.split()
                query = parts[1]
                given = parts[3:]
                evidence = {}
                for ev in given:
                    k, v = ev.split('=')
                    evidence[k] = v
                if cmd.startswith('xquery'):
                    result = exact_inference(query, evidence, vars, domains, cpts)
                elif cmd.startswith('rquery'):
                    result = rejection_sample(query, evidence, vars, domains, cpts, 1000)
                else:
                    result = gibbs_sample(query, evidence, vars, domains, cpts, 1000)
                print(result)
            elif cmd == 'help':
                print("Commands: load FILE, xquery Q | X=val ..., rquery Q | X=val ..., gquery Q | X=val ..., quit")
            else:
                print("Unknown command")
        except Exception as e:
            print("Error:", e)

if __name__ == '__main__':
    repl()
