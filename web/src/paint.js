// Slicer triangle paint (Bambu `paint_color`). Port of paint.py (decode + tessellate).
// See paint.py for the bitstream layout.

class BitReader {
  constructor(s) {
    this.s = s.trim();
    this.pos = 0; // bit position, counted from the END of the hex string
  }
  read(nbits) {
    let v = 0;
    for (let i = 0; i < nbits; i++, this.pos++) {
      const ci = this.s.length - 1 - (this.pos >> 2);
      if (ci < 0) continue; // running off the end reads zeros
      const nib = parseInt(this.s[ci], 16);
      v |= ((nib >> (this.pos & 3)) & 1) << i;
    }
    return v;
  }
}

function readNode(r) {
  const split = r.read(2);
  if (split) {
    const special = r.read(2);
    const rev = [];
    for (let i = 0; i <= split; i++) rev.push(readNode(r));
    return { split, special, children: rev.reverse() };
  }
  let state = r.read(2);
  if (state === 3) {
    for (;;) {
      const nib = r.read(4);
      state += nib;
      if (nib !== 15) break;
    }
  }
  return { split: 0, state };
}

export function decode(s) {
  return s ? readNode(new BitReader(s)) : { split: 0, state: 0 };
}

const mid = (a, b) => [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];

function split(n, v) {
  const j = n.special;
  const a = v[j % 3], b = v[(j + 1) % 3], c = v[(j + 2) % 3];
  if (n.split === 1) {
    const m = mid(b, c);
    return [[a, b, m], [m, c, a]];
  }
  if (n.split === 2) {
    const mab = mid(a, b), mac = mid(a, c);
    return [[a, mab, mac], [mab, b, mac], [b, c, mac]];
  }
  const mab = mid(a, b), mbc = mid(b, c), mca = mid(c, a);
  return [[a, mab, mca], [mab, b, mbc], [mbc, c, mca], [mab, mbc, mca]];
}

/**
 * Expand painted faces into sub-triangles.
 * Returns { tris: Float64Array (N*9), states: Int32Array (N) } (state 0 = part default).
 */
export function tessellate(vertices, faces, paint) {
  const tris = [];
  const states = [];
  const V = (k) => [vertices[3 * k], vertices[3 * k + 1], vertices[3 * k + 2]];
  for (let fi = 0; fi < paint.length; fi++) {
    const a = faces[3 * fi], b = faces[3 * fi + 1], c = faces[3 * fi + 2];
    const s = paint[fi];
    if (!s) {
      tris.push(...V(a), ...V(b), ...V(c));
      states.push(0);
      continue;
    }
    const stack = [[decode(s), [V(a), V(b), V(c)]]];
    while (stack.length) {
      const [node, t] = stack.pop();
      if (!node.split) {
        tris.push(...t[0], ...t[1], ...t[2]);
        states.push(node.state);
      } else {
        const ch = split(node, t);
        for (let i = 0; i < ch.length; i++) stack.push([node.children[i], ch[i]]);
      }
    }
  }
  return { tris: Float64Array.from(tris), states: Int32Array.from(states) };
}
