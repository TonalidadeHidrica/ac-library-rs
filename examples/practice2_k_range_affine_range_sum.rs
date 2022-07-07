use ac_library_rs::{LazySegtree, MapMonoid, ModInt998244353, Monoid};
use std::io::Read;

type Mint = ModInt998244353;
#[derive(Default)]
struct Sum;
impl Monoid for Sum {
    type S = (Mint, usize);

    fn identity(&self) -> Self::S {
        (0.into(), 0)
    }

    fn binary_operation(&self, &(a, n): &Self::S, &(b, m): &Self::S) -> Self::S {
        (a + b, n + m)
    }
}
#[derive(Default)]
struct Affine;
impl MapMonoid for Affine {
    type S = (Mint, usize);
    type F = (Mint, Mint);

    fn identity_map(&self) -> Self::F {
        (1.into(), 0.into())
    }

    fn mapping(&self, &(a, b): &Self::F, &(x, n): &Self::S) -> Self::S {
        (a * x + b * Mint::new(n), n)
    }

    // a(cx + d) + b = (ac)x + (ad+b)
    fn composition(&self, &(a, b): &Self::F, &(c, d): &Self::F) -> Self::F {
        (a * c, a * d + b)
    }
}

#[allow(clippy::many_single_char_names)]
fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let mut input = buf.split_whitespace();

    let n = input.next().unwrap().parse().unwrap();
    let q = input.next().unwrap().parse().unwrap();
    let mut segtree: LazySegtree<Sum, Affine> = input
        .by_ref()
        .take(n)
        .map(|s| (s.parse().unwrap(), 1))
        .collect::<Vec<_>>()
        .into();
    for _ in 0..q {
        match input.next().unwrap().parse().unwrap() {
            0 => {
                let l = input.next().unwrap().parse().unwrap();
                let r = input.next().unwrap().parse().unwrap();
                let b = input.next().unwrap().parse().unwrap();
                let c = input.next().unwrap().parse().unwrap();
                segtree.apply_range(l, r, (b, c));
            }
            1 => {
                let l = input.next().unwrap().parse().unwrap();
                let r = input.next().unwrap().parse().unwrap();
                println!("{}", segtree.prod(l, r).0);
            }
            _ => {}
        }
    }
}
