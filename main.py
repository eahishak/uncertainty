# CSC242: Project 3 - Uncertainty Modeling
# Date: April 3, 2025

import sys
import random
import math

# Parse .bn file format
def parse_network(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    i = 0
    num_vars = int(lines[i])
    i += 1

    vars = []
    domains = {}
    while len(vars) < num_vars:
        parts = lines[i].split()
        var = parts[0]
        domains[var] = parts[1:]
        vars.append(var)
        i += 1

    num_cpts = int(lines[i])
    i += 1
    cpts = {}

    while i < len(lines):
        header = lines[i].split()
        i += 1

        child = header[0]
        parents = header[1:] if len(header) > 1 else []

        table = []
        while i < len(lines) and lines[i] and not lines[i].startswith('#'):
            table.append(list(map(float, lines[i].split())))
            i += 1
        i += 1

        cpts[child] = {'parents': parents, 'table': table}

    return vars, domains, cpts

# Compute joint probability

def joint_prob(assignment, vars, domains, cpts):
    p = 1.0
    for var in vars:
        val = assignment[var]
        val_idx = domains[var].index(val)
        parents = cpts[var]['parents']
        table = cpts[var]['table']

        if not parents:
            probs = table[0]
        else:
            parent_vals = [assignment[p] for p in parents]
            index = 0
            stride = 1
            for p in reversed(parents):
                index += domains[p].index(assignment[p]) * stride
                stride *= len(domains[p])
            probs = table[index]

        p *= probs[val_idx]
    return p

# Exact inference

def exact_inference(query, evidence, vars, domains, cpts):
    hidden = [v for v in vars if v != query and v not in evidence]
    counts = [0.0] * len(domains[query])

    def recur(i, partial):
        if i == len(hidden):
            for j, val in enumerate(domains[query]):
                partial[query] = val
                counts[j] += joint_prob(partial, vars, domains, cpts)
        else:
            var = hidden[i]
            for val in domains[var]:
                partial[var] = val
                recur(i+1, partial)

    recur(0, dict(evidence))
    total = sum(counts)
    return [round(x/total, 4) for x in counts]

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
                index = 0
                stride = 1
                for p in reversed(parents):
                    index += domains[p].index(sample[p]) * stride
                    stride *= len(domains[p])
                probs = table[index]

            r = random.random()
            total = 0.0
            for i, prob in enumerate(probs):
                total += prob
                if r <= total:
                    sample[var] = domains[var][i]
                    break

            if var in evidence and sample[var] != evidence[var]:
                consistent = False
                break

        if consistent:
            idx = domains[query].index(sample[query])
            counts[idx] += 1

    total = sum(counts)
    return [round(x/total, 4) if total > 0 else 0.0 for x in counts]

# Gibbs sampling

def gibbs_sample(query, evidence, vars, domains, cpts, N, burn=100):
    state = dict(evidence)
    hidden = [v for v in vars if v not in evidence]
    for var in hidden:
        state[var] = random.choice(domains[var])

    counts = [0.0] * len(domains[query])

    for i in range(N + burn):
        for var in hidden:
            dist = []
            for val in domains[var]:
                state[var] = val
                dist.append(joint_prob(state, vars, domains, cpts))
            total = sum(dist)
            r = random.random()
            acc = 0
            for j, val in enumerate(domains[var]):
                acc += dist[j] / total
                if r <= acc:
                    state[var] = val
                    break

        if i >= burn:
            idx = domains[query].index(state[query])
            counts[idx] += 1

    total = sum(counts)
    return [round(x/total, 4) if total > 0 else 0.0 for x in counts]

# REPL interface

def repl():
    vars, domains, cpts = [], {}, {}
    print("BayesNet REPL. Type 'help' for commands.")
    while True:
        try:
            line = input("BN > ").strip()
            if not line:
                continue
            if line == 'quit':
                break
            elif line.startswith('load'):
                fname = line.split(maxsplit=1)[1]
                vars, domains, cpts = parse_network(fname)
                print("Network loaded.")
            elif line.startswith(('xquery', 'rquery', 'gquery')):
                parts = line.split('|')
                left = parts[0].split()
                query = left[1]
                evidence = {}
                if len(parts) > 1:
                    for pair in parts[1].strip().split():
                        k, v = pair.split('=')
                        evidence[k] = v
                if line.startswith('xquery'):
                    print(exact_inference(query, evidence, vars, domains, cpts))
                elif line.startswith('rquery'):
                    print(rejection_sample(query, evidence, vars, domains, cpts, 1000))
                else:
                    print(gibbs_sample(query, evidence, vars, domains, cpts, 1000))
            elif line == 'help':
                print("Commands:")
                print("  load FILE")
                print("  xquery Q | X=val ...")
                print("  rquery Q | X=val ...")
                print("  gquery Q | X=val ...")
                print("  quit")
            else:
                print("Unknown command")
        except Exception as e:
            print("Error:", e)

if __name__ == '__main__':
    repl()
