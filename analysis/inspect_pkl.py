"""
Reads the structure of a torch-pickled .pkl file WITHOUT requiring torch to
be installed, by stubbing out torch's tensor-rebuild classes. Used to audit
this repo's data/vectors/*.pkl artifacts (see defect_report.md).

CLI usage:
    python3 analysis/inspect_pkl.py data/vectors/some_file.pkl [more files...]
    -> prints the structure (keys, shapes, dtypes) of each file.

Library usage (see audit.py):
    from inspect_pkl import StubUnpickler
    with open(path, 'rb') as f:
        obj = StubUnpickler(f).load()
"""
import pickle, io, sys, json

class Stub:
    def __init__(self, name): self.name = name
    def __call__(self, *a, **k):
        if self.name.endswith('_rebuild_tensor_v2'):
            storage, offset, size, stride = a[0], a[1], a[2], a[3]
            return {'__tensor__': True, 'shape': list(size), 'stride': list(stride),
                    'offset': offset, 'storage': storage}
        if self.name.endswith('_load_from_bytes'):
            return {'__storage_bytes__': len(a[0]), 'raw': a[0]}
        return {'__call__': self.name, 'args': [repr(x)[:80] for x in a]}

class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('torch'):
            return Stub(module + '.' + name)
        return super().find_class(module, name)

def describe(o, depth=0):
    pad = '  ' * depth
    if isinstance(o, dict):
        if o.get('__tensor__'):
            return f"Tensor shape={o['shape']} storage_bytes={o['storage'].get('__storage_bytes__')}"
        lines = []
        for k, v in o.items():
            if k == 'raw': continue
            lines.append(f"{pad}{k!r}: {describe(v, depth+1)}")
        return "{\n" + "\n".join(lines) + f"\n{pad}}}"
    if isinstance(o, list):
        if len(o) > 4:
            return f"list[{len(o)}] e.g. {o[:3]!r} ... {o[-1]!r}"
        return repr(o)
    return repr(o)[:200]

if __name__ == '__main__':
    for path in sys.argv[1:]:
        print("=" * 70)
        print(path.split('/')[-1])
        print("=" * 70)
        with open(path, 'rb') as f:
            obj = StubUnpickler(f).load()
        print(describe(obj))
        print()
