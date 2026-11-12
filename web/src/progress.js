// Progress reporting: a Span is a slice [a, b] of the overall bar. Steps split
// their span into weighted sub-spans, so nested work (parts -> colors -> steps)
// maps onto one monotone 0..1 value. Weights are rough cost estimates; steps
// that report their own progress (marching blocks) fill their slice smoothly.

let sink = null;
export const setProgressSink = (fn) => (sink = fn);

export class Span {
  constructor(a = 0, b = 1) {
    this.a = a;
    this.b = b;
  }
  sub(a, b) {
    const w = this.b - this.a;
    return new Span(this.a + w * a, this.a + w * b);
  }
  /** Consecutive sub-spans proportional to weights. */
  split(weights) {
    const total = weights.reduce((s, w) => s + w, 0) || 1;
    let acc = 0;
    return weights.map((w) => {
      const s = this.sub(acc / total, (acc + w) / total);
      acc += w;
      return s;
    });
  }
  at(f, label) {
    sink?.(this.a + (this.b - this.a) * Math.min(1, Math.max(0, f)), label);
  }
}
