"""Query-prompt adapter for the unchanged official GraphRouter network."""
import math

import torch

from train_router_support_graph import MODELS, PROMPTS, TASKS, vector


def action_data(rows, prompts, pool, lo, span):
    """Outcomes for explicitly requested actions only; never scorer inputs."""
    if not math.isfinite(lo) or not math.isfinite(span) or span <= 0:
        raise ValueError('invalid fixed monetary reference')
    values = []
    for row in rows:
        edges = {(e['prompt'], e['model']): e for e in row['action_edges']}
        actions = []
        for prompt in prompts:
            models = []
            for model in pool:
                e = edges[prompt, model]
                p, money, cq = (float(e[k]) for k in ('performance', 'cost_proxy_money', 'cost_norm_query'))
                if not all(math.isfinite(v) for v in (p, money, cq)) or not (0 <= p <= 1 and money >= 0 and 0 <= cq <= 1):
                    raise ValueError('invalid P/money/qnorm outcome; no cost fallback')
                models.append([p, (money-lo)/span, cq])
            actions.append(models)
        values.append(actions)
    return torch.tensor(values, dtype=torch.float32)


def cost_reference(fit, seen):
    # Intentionally Direct only, even though the joint graph has four prompts.
    costs = [float(e['cost_proxy_money']) for r in fit for e in r['action_edges']
             if e['prompt'] == 'direct' and e['model'] in seen]
    if not costs or not all(math.isfinite(c) and c >= 0 for c in costs):
        raise ValueError('invalid known Direct monetary costs')
    lo, hi = min(costs), max(costs)
    if hi <= lo:
        raise ValueError('degenerate known Direct monetary reference')
    return lo, hi


def best_action_labels(values, lam):
    reward = values[..., 0]-lam*values[..., 1]
    best = reward.flatten(1).argmax(1)
    labels = torch.nn.functional.one_hot(best, num_classes=reward.shape[1]*reward.shape[2])
    return best, labels.flatten().float()


def make_visible(context, prompts, pool, values, lam):
    if values.shape != (len(context), len(prompts), len(pool), 3):
        raise ValueError('outcome alignment mismatch')
    return [(r['qid'], p, m, [float(values[i,j,k,0]-lam*values[i,j,k,1]),
                              float(values[i,j,k,1]), float(values[i,j,k,0])])
            for i,r in enumerate(context) for j,p in enumerate(prompts) for k,m in enumerate(pool)]


def graph_inputs(context, targets, prompts, pool, tables, visible, device):
    """No raw outcomes read here. Visible edges must originate in context."""
    if not prompts or not pool or len(set(prompts)) != len(prompts) or len(set(pool)) != len(pool):
        raise ValueError('empty or duplicated candidate pool')
    rows = context+targets
    if len({r['qid'] for r in rows}) != len(rows):
        raise ValueError('context/target overlap or duplicate qids')
    qt, tt, pt, mt = tables
    sources = [(r, p) for r in rows for p in prompts]
    query = torch.stack([torch.cat((vector(qt,r['qid'],()), vector(pt,p,PROMPTS))) for r,p in sources]).to(device)
    task = torch.stack([vector(tt,r['task'],TASKS) for r,p in sources]).to(device)
    models = torch.stack([vector(mt,m,MODELS) for m in pool]).to(device)
    if task.shape[1] != models.shape[1]:
        raise ValueError('official FeatureAlign requires equal task/model input widths')
    index = {(r['qid'],p): i for i,(r,p) in enumerate(sources)}
    context_ids = {r['qid'] for r in context}
    src, dst, attrs, seen = [], [], [], set()
    for qid,p,m,attr in visible:
        key = (qid,p,m)
        if qid not in context_ids or (qid,p) not in index or m not in pool or key in seen:
            raise ValueError('visible edge outside context/pool or duplicated')
        if len(attr) != 3 or not all(math.isfinite(float(x)) for x in attr):
            raise ValueError('invalid visible edge attributes')
        seen.add(key)
        src.append(index[qid,p]); dst.append(len(sources)+pool.index(m)); attrs.append(attr)
    return task, query, models, src, dst, attrs


def evaluate_scores(net, context, targets, prompts, pool, tables, visible, device):
    task, query, models, src, dst, attrs = graph_inputs(context, targets, prompts, pool, tables, visible, device)
    count = len(src)
    for i in range(len(context)*len(prompts), len(query)):
        for m in range(len(pool)):
            src.append(i); dst.append(len(query)+m)
    edges = torch.tensor([src,dst], dtype=torch.long, device=device)
    attributes = torch.zeros(len(src), 3, device=device)
    if count:
        attributes[:count] = torch.tensor(attrs, dtype=torch.float32, device=device)
    can_see = torch.arange(len(src),device=device) < count
    net.eval()
    with torch.inference_mode():
        scores = net(task,query,models,edges,edge_mask=~can_see,
                     edge_can_see=can_see,edge_weight=attributes)
    result = scores.reshape(len(targets),len(prompts),len(pool)).cpu()
    if not torch.isfinite(result).all():
        raise RuntimeError('nonfinite candidate scores')
    return result
