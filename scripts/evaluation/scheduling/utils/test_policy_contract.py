#!/usr/bin/env python3
"""Every policy that can report its actions must declare it.

`main.py::_run_calls` dispatches on `policy.returns_actions`. Policies that declare it are
re-scored with the cross-cluster cache-warmup penalty, get a row in diagnostics.csv, and can be
action-dumped. Policies that do not are used verbatim -- no warmup charge.

So a migrating policy that omits the attribute runs a race its competitors pay for. That is not
hypothetical: make_hetero_reactive_fallback_gate omitted it, went 55,048 migrations unpaid, and
its ED2P score sat below the global oracle. See POLICIES.md.

Run: python3 utils/test_policy_contract.py
"""
import inspect, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import scheduling_policies as sched   # noqa: E402
import dvfs_policies as dvfs          # noqa: E402


def build(fn):
    """Instantiate a factory with whatever minimal arguments it needs, or return None."""
    for args in ((), (1,), (1.0,), ('P',)):
        try:
            return fn(*args)
        except TypeError:
            continue
        except Exception:
            return None
    return None


def main():
    bad, checked = [], 0
    for mod in (sched, dvfs):
        for name in dir(mod):
            if not name.startswith('make_'):
                continue
            p = build(getattr(mod, name))
            if p is None or not callable(p):
                continue
            sig = inspect.signature(p)
            if '_return_actions' not in sig.parameters:
                continue            # cannot report actions; nothing to declare
            checked += 1
            if not getattr(p, 'returns_actions', False):
                bad.append(f'{mod.__name__}.{name}')
    print(f'checked {checked} policy factories that accept _return_actions')
    if bad:
        print('\nFAIL: these accept _return_actions but do not set policy.returns_actions = True')
        print('      -> they are scored WITHOUT the warmup penalty and are missing from diagnostics:')
        for b in bad:
            print(f'        {b}')
        sys.exit(1)
    print('PASS: every action-capable policy declares returns_actions')


if __name__ == '__main__':
    main()
