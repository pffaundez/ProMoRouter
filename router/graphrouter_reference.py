"""Public-data reference for GraphRouter's reconstructed few-shot experiment.

No ProMoRouter logs, prompt embeddings or custom scorer are used.
"""
import ast
import csv
import hashlib
import io
import json
import math
import pickle
import random
import re
from pathlib import Path

import numpy as np
import torch

UPSTREAM = '0edfb8758d45acd93c8029354867ffb49bb8a97f'
DRIVE_ID = '1YYn-BV-5s2amh6mKLqKMR0H__JB-CKU4'
RELEASE_CSV_SHA256 = 'df6fd3e7cd932b85801d8cb4d782188250266dad5f53eff5ed95253134a6349c'
BLOBS = {
    'LLM_Descriptions.json': '4a77bd97d65ab4a76cdaeb62d17ed4974a9ec42d',
    'llm_description_embedding.pkl': 'b54d7efc56d44ce938ff80033aeb4d9fb34ec47c',
    'config.yaml': '36e4b54b6f9ad6e3586d9e45fca5a72157398edd',
}
MODELS = ('LLaMA-3 (8b)', 'Mixtral-8x7B', 'NousResearch', 'LLaMA-2 (7b)',
          'Mistral-7b', 'LLaMA-3 (70b)', 'LLaMA-3-Turbo (8b)',
          'LLaMA-3-Turbo (70b)', 'Llama-3.1-Turbo (70b)', 'Qwen-1.5 (72b)')
TASKS = ('alpaca_data', 'GSM8K', 'multi_news', 'SQUAD')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_blob(path):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if digest != BLOBS[path.name]:
        raise ValueError(f'upstream artifact checksum mismatch: {path.name}')
    return raw


class NumpyArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy', 'ndarray'), ('numpy', 'dtype'),
                   ('numpy.core.multiarray', '_reconstruct'),
                   ('numpy._core.multiarray', '_reconstruct'),
                   ('numpy.core.numeric', '_frombuffer'),
                   ('numpy._core.numeric', '_frombuffer')}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'unexpected upstream pickle class: {module}.{name}')
        return super().find_class(module, name)


def embedding(text):
    try:
        obj = ast.literal_eval(text)
        arr = np.asarray(obj, dtype=np.float32).reshape(-1)
    except (ValueError, SyntaxError, TypeError):
        # NumPy's whitespace-rendered [[...]] arrays are not JSON/Python lists.
        pattern = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
        stripped = re.sub(pattern, '', text)
        if re.sub(r'[\s,\[\]]', '', stripped):
            raise ValueError('invalid serialized embedding')
        arr = np.asarray([float(x) for x in re.findall(pattern, text)], dtype=np.float32)
    if not len(arr) or not np.isfinite(arr).all():
        raise ValueError('empty/nonfinite embedding')
    return torch.from_numpy(arr.copy())


def read_public_csv(path, models=MODELS, allow_upstream_blank_label=False):
    """Preserve the upstream query/model order, rejecting ambiguous alignment."""
    csv.field_size_limit(2**31-1)
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'task_id','query','query_embedding','task_description_embedding','llm','effect','cost'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('not the authors\' router_data.csv; required columns missing')
        source = list(reader)
    if not source or len(source)%len(models):
        raise ValueError('incomplete query/model groups')
    queries, tasks, names, perf, cost = [], [], [], [], []
    blank_resolutions = 0
    for start in range(0,len(source),len(models)):
        group = source[start:start+len(models)]
        labels = [r['llm'] for r in group]
        if allow_upstream_blank_label and tuple(models)==MODELS and labels[3]=='':
            labels[3]=MODELS[3];blank_resolutions+=1
        if labels != list(models):
            raise ValueError(f'model order/coverage mismatch at CSV row {start+2}')
        first = group[0]
        if any((r['query'],r['task_id']) != (first['query'],first['task_id']) for r in group):
            raise ValueError('query/model block mixes queries or tasks')
        q, t = embedding(first['query_embedding']), embedding(first['task_description_embedding'])
        for r in group[1:]:
            if not torch.equal(q,embedding(r['query_embedding'])) or not torch.equal(t,embedding(r['task_description_embedding'])):
                raise ValueError('inconsistent embeddings within a query block')
        p = [float(r['effect']) for r in group]; c = [float(r['cost']) for r in group]
        if not all(math.isfinite(v) and 0<=v<=1 for v in p+c):
            raise ValueError('published effect/cost must be finite in [0,1]; no renormalization')
        queries.append(q); tasks.append(t); names.append(first['task_id']); perf.append(p); cost.append(c)
    return dict(query=torch.stack(queries),task=torch.stack(tasks),task_names=names,
                performance=torch.tensor(perf),cost=torch.tensor(cost),
                positional_blank_model_labels_resolved=blank_resolutions)


def partitions(task_names, seed=1881, support_size=80):
    order = list(dict.fromkeys(task_names))
    groups = {t:[i for i,name in enumerate(task_names) if name==t] for t in order}
    sizes = {len(g) for g in groups.values()}
    if len(sizes)!=1:
        raise ValueError('upstream split assumes equal task populations')
    train, val, test = [], [], []
    offset = 0
    for t in order:
        ids = groups[t]
        if ids != list(range(offset,offset+len(ids))):
            raise ValueError('upstream split assumes contiguous task blocks')
        offset += len(ids)
        a, b = int(len(ids)*.7), int(len(ids)*.1)
        train.extend(ids[:a]); val.extend(ids[a:a+b]); test.extend(ids[a+b:])
    if not train or not val or not test or support_size>len(train):
        raise ValueError('insufficient queries for splits/support')
    support = sorted(random.Random(seed).sample(train,support_size))
    return dict(train=train,validation=val,test=test,support=support)


def prepare(root, synthetic=False):
    root = Path(root)
    descriptions = json.loads(verify_blob(root/'LLM_Descriptions.json'))
    if list(descriptions) != list(MODELS):
        raise ValueError('unexpected upstream model order')
    raw = verify_blob(root/'llm_description_embedding.pkl')
    models = NumpyArrayUnpickler(io.BytesIO(raw)).load()
    models = torch.as_tensor(np.asarray(models,dtype=np.float32).copy())
    if models.ndim!=2 or models.shape[0]!=10 or not torch.isfinite(models).all():
        raise ValueError('invalid upstream model embeddings')
    verify_blob(root/'config.yaml')
    released = sha256(root/'router_data.csv') == RELEASE_CSV_SHA256
    if not released and not synthetic:
        raise ValueError("CSV differs from the verified authors' release; refusing another dataset under this reference")
    data = read_public_csv(root/'router_data.csv',allow_upstream_blank_label=released)
    counts = {t:data['task_names'].count(t) for t in dict.fromkeys(data['task_names'])}
    publication_shape = set(counts)==set(TASKS) and all(n==600 for n in counts.values())
    report = dict(upstream_commit=UPSTREAM,source_csv_sha256=sha256(root/'router_data.csv'),
                  upstream_artifact_sha256={n:sha256(root/n) for n in BLOBS},
                  task_counts=counts,paper_population_match=publication_shape,
                  authors_release_verified=released,
                  released_interaction_rows_per_task={t:n*10 for t,n in counts.items()},
                  paper_table_cases_per_task=600,
                  paper_cases_unit_ambiguity='table says 600 Cases/task; released data has 60 query blocks and 600 interaction rows/task',
                  blank_model_label_resolution=dict(count=data['positional_blank_model_labels_resolved'],
                      rule='only empty fourth position maps to LLaMA-2 (7b), as positional upstream code; all other model labels verified'),
                  model_names=list(MODELS),known_models=list(MODELS[:6]),unseen_models=list(MODELS[6:]),
                  mode='synthetic_engineering' if synthetic else 'public_code_fewshot_reconstruction',
                  quantitative_paper_reproduction_verified=False,
                  cost='authors\' stored task-normalized cost; unchanged',
                  reward='0.5*effect - 0.5*cost (Balance)',
                  differences=['public code has GeneralConv width16/AdamW constant lr, paper describes GAT width32/Adam schedule',
                               'public code has no explicit unseen-model driver; this reconstructs the published 6/4 +80 protocol',
                               'support draw is reconstructed using config seed1881; original auxiliary query IDs not released',
                               'primary evaluation uses final epoch as public test(); best-validation checkpoint saved separately',
                               'full-query embeddings enter training BatchNorm as in public code; no validation/test outcomes do'])
    (root/'data_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not publication_shape and not released and not synthetic:
        raise ValueError('unverified dataset population; no fallback to ProMoRouter')
    if data['task'].shape[1]!=models.shape[1]:
        raise ValueError('task/model widths do not satisfy official FeatureAlign')
    data.update(model=models,metadata=report,splits=partitions(data['task_names']))
    return data


def graph(data, pool, visible_queries, visible_models, predict_queries, device):
    """Visible outcome lookup is restricted to explicitly permitted q/model pairs."""
    q = data['query'].to(device); t = data['task'].to(device)
    m = data['model'][list(pool)].to(device)
    nq = len(q)
    src, dst, attrs = [], [], []
    if len(visible_queries)!=len(visible_models):
        raise ValueError('visible-query/model list mismatch')
    observed_edges=set()
    for qi,mis in zip(visible_queries,visible_models):
        for mi in mis:
            if mi not in pool or qi in predict_queries or (qi,mi) in observed_edges:
                raise ValueError('visible edge overlaps target or candidate outside pool')
            observed_edges.add((qi,mi))
            p,c = float(data['performance'][qi,mi]),float(data['cost'][qi,mi])
            if not math.isfinite(p+c):
                raise ValueError('nonfinite visible outcome')
            src.append(qi);dst.append(nq+pool.index(mi));attrs.append([.5*p-.5*c,c,p])
    count = len(src)
    for qi in predict_queries:
        for j in range(len(pool)):
            src.append(qi);dst.append(nq+j);attrs.append([0.,0.,0.])
    edges = torch.tensor([src,dst],dtype=torch.long,device=device)
    attr = torch.tensor(attrs,dtype=torch.float32,device=device).reshape(-1,3)
    see = torch.arange(len(src),device=device)<count
    return t,q,m,edges,attr,see


def predict(net, data, pool, visible_queries, visible_models, targets, device):
    t,q,m,e,a,see = graph(data,pool,visible_queries,visible_models,targets,device)
    net.eval()
    with torch.inference_mode():
        scores = net(t,q,m,e,edge_mask=~see,edge_can_see=see,edge_weight=a)
    return scores.reshape(len(targets),len(pool)).cpu()
