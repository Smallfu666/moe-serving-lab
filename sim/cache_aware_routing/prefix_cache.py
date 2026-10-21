"""Page-granular radix prefix cache with LRU leaf eviction, modelled on SGLang's RadixCache.

A sequence is a chain of page keys: key i hashes key i-1 together with the tokens of page i, so equal keys mean equal
prefixes. Only whole pages are cached (SGLang frees the partial last page). Nodes referenced by a running request are
locked and never evicted; eviction removes the least recently used unlocked leaf, as SGLang's `evict` does.
"""

import heapq
import itertools


def page_keys(tokens, page):
    keys, h = [], 0
    for i in range(len(tokens) // page):
        h = hash((h, tuple(tokens[i * page:(i + 1) * page])))
        keys.append(h)
    return keys


class Node:
    __slots__ = ("parent", "children", "lock", "last")

    def __init__(self, parent, now):
        self.parent, self.children, self.lock, self.last = parent, 0, 0, now


class PrefixCache:
    def __init__(self, capacity_tokens, page):
        self.page = page
        self.cap = capacity_tokens // page
        self.nodes = {}
        self.heap = []
        self.tick = itertools.count()
        self.overflow_pages = 0

    def match(self, keys):
        n = 0
        for k in keys:
            if k not in self.nodes:
                break
            n += 1
        return n

    def _push(self, k):
        nd = self.nodes[k]
        if nd.children == 0 and nd.lock == 0:
            heapq.heappush(self.heap, (nd.last, next(self.tick), k))

    def evict(self, need):
        """Evict LRU unlocked leaves until `need` pages are free. Returns pages still missing."""
        while len(self.nodes) + need > self.cap and self.heap:
            last, _, k = heapq.heappop(self.heap)
            nd = self.nodes.get(k)
            if nd is None or nd.children or nd.lock or nd.last != last:
                continue
            del self.nodes[k]
            if nd.parent is not None:
                p = self.nodes[nd.parent]
                p.children -= 1
                self._push(nd.parent)
        return max(0, len(self.nodes) + need - self.cap)

    def admit(self, keys, now, strict=False):
        """Lock the cached prefix of `keys`, evict for the rest, insert and lock it. Returns matched pages.

        When eviction cannot make room: strict returns None and leaves the cache locked as before (the caller queues
        the request); otherwise the pages are inserted over capacity and counted in `overflow_pages`.
        """
        hit = self.match(keys)
        for k in keys[:hit]:
            nd = self.nodes[k]
            nd.lock += 1
            nd.last = now
        missing = self.evict(len(keys) - hit)
        if missing and strict:
            self.release(keys[:hit], now)
            return None
        self.overflow_pages += missing
        parent = keys[hit - 1] if hit else None
        for k in keys[hit:]:
            self.nodes[k] = Node(parent, now)
            self.nodes[k].lock = 1
            if parent is not None:
                self.nodes[parent].children += 1
            parent = k
        return hit

    def release(self, keys, now):
        for k in keys:
            nd = self.nodes[k]
            nd.lock -= 1
            nd.last = now
        for k in keys:
            self._push(k)

    def free_pages(self):
        return self.cap - len(self.nodes)
