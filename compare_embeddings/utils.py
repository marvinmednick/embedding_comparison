import re

SEARCH_AREA = 0.2
SPLITTERS = [
    "\n\n",
    "\n \n",
    ":\n",
    ": ",
    "\\.\n",
    "\\. ",
    ";\n",
    "; ",
    ",\n",
    ", ",
    "\\.",
    ",",
    "\n",
    " ",
]


def find_best_split_point(astr, reverse_splitters=False):
    center = len(astr) // 2

    if reverse_splitters:
        splitters = SPLITTERS[::-1]
        nearby = int(round(len(astr) * SEARCH_AREA / 4))
    else:
        splitters = SPLITTERS
        nearby = int(round(len(astr) * SEARCH_AREA))

    for splitter in splitters:
        starts = [m.start() for m in re.finditer(splitter, astr)]
        starts.sort(key=lambda x: abs(x - center))

        if starts and abs(starts[0] - center) < nearby:
            return starts[0] + len(splitter.replace("\\.", "."))

    return center


def shorten_text(astr, max_length):
    if len(astr) < max_length:
        return astr
    else:
        str_sample = astr[: int(round(max_length / (0.5 + SEARCH_AREA)))]
        spoint = find_best_split_point(str_sample, reverse_splitters=True)
        return str_sample[:spoint]


def create_size_buckets(sizes):

    size_buckets = {size: 0 for size in sizes}
    size_buckets['size_list'] = sizes
    size_buckets['max'] = 0  # For sizes larger than the largest specified size
    return size_buckets


def increment_bucket(size_buckets, item_size):
    for size in sorted(size_buckets['size_list']):
        if item_size < size:
            size_buckets[size] += 1
            break
    else:
        size_buckets['max'] += 1
    return size_buckets


