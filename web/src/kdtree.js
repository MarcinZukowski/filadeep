// Static 3D KD-tree for nearest-neighbour distance queries with an upper bound
// (what scipy's cKDTree.query(..., distance_upper_bound=ub) gives us in Python).
//
// Layout: nodes in flat typed arrays, each with its bounding box (for pruning)
// and a [start, end) range into the point arrays; points are reordered into
// leaf order (separate x/y/z arrays), so a leaf scan reads contiguous memory.

export class KDTree {
  /** pts: Float64Array of xyz triples. */
  constructor(pts, leafSize = 16) {
    const n = pts.length / 3;
    this.n = n;
    const idx = new Uint32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    const maxNodes = Math.max(1, 2 * Math.ceil(n / leafSize) * 2);
    this.box = new Float64Array(maxNodes * 6);
    this.left = new Int32Array(maxNodes).fill(-1);
    this.right = new Int32Array(maxNodes).fill(-1);
    this.start = new Uint32Array(maxNodes);
    this.end = new Uint32Array(maxNodes);
    let count = 0;

    const build = (s, e) => {
      const node = count++;
      this.start[node] = s;
      this.end[node] = e;
      let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
      for (let i = s; i < e; i++) {
        const k = 3 * idx[i];
        const x = pts[k], y = pts[k + 1], z = pts[k + 2];
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (z < z0) z0 = z; if (z > z1) z1 = z;
      }
      const b = node * 6;
      this.box[b] = x0; this.box[b + 1] = y0; this.box[b + 2] = z0;
      this.box[b + 3] = x1; this.box[b + 4] = y1; this.box[b + 5] = z1;
      if (e - s <= leafSize) return node;
      const dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
      const dim = dx >= dy && dx >= dz ? 0 : dy >= dz ? 1 : 2;
      const m = (s + e) >> 1;
      select(idx, pts, dim, s, e - 1, m);
      this.left[node] = build(s, m);
      this.right[node] = build(m, e);
      return node;
    };
    if (n > 0) build(0, n);
    this.nodes = count;
    this.x = new Float64Array(n);
    this.y = new Float64Array(n);
    this.z = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const k = 3 * idx[i];
      this.x[i] = pts[k];
      this.y[i] = pts[k + 1];
      this.z[i] = pts[k + 2];
    }
    this.stack = new Int32Array(256);
    this.stackD = new Float64Array(256);
  }

  /** Distance to the nearest point, or Infinity if none is closer than ub. */
  query(qx, qy, qz, ub = Infinity) {
    if (this.n === 0) return Infinity;
    let best = ub * ub;
    const { box, left, right, start, end, x, y, z, stack, stackD } = this;
    let sp = 0;
    stack[sp] = 0;
    stackD[sp++] = 0;
    while (sp > 0) {
      sp--;
      if (stackD[sp] >= best) continue;
      const node = stack[sp];
      const l = left[node];
      if (l < 0) {
        for (let i = start[node], e = end[node]; i < e; i++) {
          const ax = x[i] - qx, ay = y[i] - qy, az = z[i] - qz;
          const d = ax * ax + ay * ay + az * az;
          if (d < best) best = d;
        }
        continue;
      }
      const r = right[node];
      const dl = boxDist(box, l, qx, qy, qz);
      const dr = boxDist(box, r, qx, qy, qz);
      // Push the farther child first so the nearer one is searched first.
      if (dl < dr) {
        if (dr < best) { stack[sp] = r; stackD[sp++] = dr; }
        if (dl < best) { stack[sp] = l; stackD[sp++] = dl; }
      } else {
        if (dl < best) { stack[sp] = l; stackD[sp++] = dl; }
        if (dr < best) { stack[sp] = r; stackD[sp++] = dr; }
      }
    }
    return best < ub * ub ? Math.sqrt(best) : Infinity;
  }
}

function boxDist(box, node, qx, qy, qz) {
  const b = node * 6;
  let d = 0, t;
  if ((t = box[b] - qx) > 0) d += t * t; else if ((t = qx - box[b + 3]) > 0) d += t * t;
  if ((t = box[b + 1] - qy) > 0) d += t * t; else if ((t = qy - box[b + 4]) > 0) d += t * t;
  if ((t = box[b + 2] - qz) > 0) d += t * t; else if ((t = qz - box[b + 5]) > 0) d += t * t;
  return d;
}

/** Quickselect: reorder idx[lo..hi] so idx[k] has the k-th smallest coordinate `dim`. */
function select(idx, pts, dim, lo, hi, k) {
  while (hi > lo) {
    const pv = pts[3 * idx[(lo + hi) >> 1] + dim];
    let i = lo, j = hi;
    while (i <= j) {
      while (pts[3 * idx[i] + dim] < pv) i++;
      while (pts[3 * idx[j] + dim] > pv) j--;
      if (i <= j) {
        const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        i++; j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return;
  }
}
