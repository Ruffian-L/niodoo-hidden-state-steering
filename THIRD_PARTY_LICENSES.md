# Third-party licenses

This project's own code is MIT (see `LICENSE`). It builds on the Rust crates below and
redistributes one Meta Llama 3.1 asset. This file is the attribution record for all of it.

Generated from `cargo metadata` over the resolved dependency graph of the `niodoo` crate —
regenerate with:
`(cd niodoo && cargo metadata --format-version 1) | python3 scripts/generate_third_party_licenses.py`
Full license texts for the common licenses are in `licenses/` (Apache-2.0) and `LICENSE`
(MIT); every crate's own text is available at its linked repository.

## Meta Llama 3.1 (redistributed asset)

`model/tokenizer.json` is part of the Llama 3.1 materials, redistributed here so the
reproduction runs out of the box. The model weights themselves are downloaded at run time
(bartowski's GGUF quantization of Meta-Llama-3.1-8B-Instruct) and are NOT in this repository.

**Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc.
All Rights Reserved.** Full text: `licenses/LLAMA-3.1-COMMUNITY-LICENSE.txt`. This project is
**Built with Llama**.

## Direct dependencies (51)

| crate | version | license |
|---|---|---|
| [anyhow](https://github.com/dtolnay/anyhow) | 1.0.102 | MIT OR Apache-2.0 |
| [axum](https://github.com/tokio-rs/axum) | 0.7.9 | MIT |
| [bincode](https://github.com/servo/bincode) | 1.3.3 | MIT |
| [bytemuck](https://github.com/Lokathor/bytemuck) | 1.25.0 | Zlib OR Apache-2.0 OR MIT |
| [candle-core](https://github.com/huggingface/candle) | 0.10.2 | MIT OR Apache-2.0 |
| [candle-nn](https://github.com/huggingface/candle) | 0.10.2 | MIT OR Apache-2.0 |
| [candle-transformers](https://github.com/huggingface/candle) | 0.10.2 | MIT OR Apache-2.0 |
| [chrono](https://github.com/chronotope/chrono) | 0.4.44 | MIT OR Apache-2.0 |
| [clap](https://github.com/clap-rs/clap) | 4.6.1 | MIT OR Apache-2.0 |
| [cudarc](https://github.com/coreylowman/cudarc) | 0.17.8 | MIT OR Apache-2.0 |
| [cudarc](https://github.com/chelsea0x3b/cudarc) | 0.19.8 | MIT OR Apache-2.0 |
| [ed25519-dalek](https://github.com/dalek-cryptography/curve25519-dalek/tree/main/ed25519-dalek) | 2.2.0 | BSD-3-Clause |
| [flate2](https://github.com/rust-lang/flate2-rs) | 1.1.9 | MIT OR Apache-2.0 |
| [friedrich](https://github.com/nestordemeure/friedrich) | 0.5.0 | Apache-2.0 |
| [futures](https://github.com/rust-lang/futures-rs) | 0.3.32 | MIT OR Apache-2.0 |
| [glam](https://github.com/bitshifter/glam-rs) | 0.27.0 | MIT OR Apache-2.0 |
| [half](https://github.com/VoidStarKat/half-rs) | 2.7.1 | MIT OR Apache-2.0 |
| [hf-hub](https://github.com/huggingface/hf-hub) | 0.3.2 | Apache-2.0 |
| [hnsw_rs](https://github.com/jean-pierreBoth/hnswlib-rs) | 0.2.1 | MIT/Apache-2.0 |
| [memmap2](https://github.com/RazrFalcon/memmap2-rs) | 0.9.11 | MIT OR Apache-2.0 |
| [nalgebra](https://github.com/dimforge/nalgebra) | 0.29.0 | BSD-3-Clause |
| [nalgebra](https://github.com/dimforge/nalgebra) | 0.30.1 | BSD-3-Clause |
| [nalgebra](https://github.com/dimforge/nalgebra) | 0.32.6 | BSD-3-Clause |
| [native-tls](https://github.com/rust-native-tls/rust-native-tls) | 0.2.18 | MIT OR Apache-2.0 |
| [ndarray](https://github.com/rust-ndarray/ndarray) | 0.15.6 | MIT OR Apache-2.0 |
| [ndarray-npy](https://github.com/jturner314/ndarray-npy) | 0.8.1 | MIT OR Apache-2.0 |
| [notify](https://github.com/notify-rs/notify.git) | 6.1.1 | CC0-1.0 |
| [rand](https://github.com/rust-random/rand) | 0.8.6 | MIT OR Apache-2.0 |
| [rand](https://github.com/rust-random/rand) | 0.9.4 | MIT OR Apache-2.0 |
| [rayon](https://github.com/rayon-rs/rayon) | 1.12.0 | MIT OR Apache-2.0 |
| [regex](https://github.com/rust-lang/regex) | 1.12.3 | MIT OR Apache-2.0 |
| [reqwest](https://github.com/seanmonstar/reqwest) | 0.11.27 | MIT OR Apache-2.0 |
| [rkyv](https://github.com/rkyv/rkyv) | 0.7.46 | MIT |
| [rusqlite](https://github.com/rusqlite/rusqlite) | 0.30.0 | MIT |
| [safetensors](https://github.com/huggingface/safetensors) | 0.4.5 | Apache-2.0 |
| [safetensors](https://github.com/huggingface/safetensors) | 0.7.0 | Apache-2.0 |
| [serde](https://github.com/serde-rs/serde) | 1.0.228 | MIT OR Apache-2.0 |
| [serde-big-array](https://github.com/est31/serde-big-array) | 0.5.1 | MIT OR Apache-2.0 |
| [serde_json](https://github.com/serde-rs/json) | 1.0.149 | MIT OR Apache-2.0 |
| [sha2](https://github.com/RustCrypto/hashes) | 0.10.9 | MIT OR Apache-2.0 |
| [shared_memory](https://github.com/elast0ny/shared_memory-rs) | 0.12.4 | MIT OR Apache-2.0 |
| [statrs](https://github.com/statrs-dev/statrs) | 0.16.1 | MIT |
| [tantivy](https://github.com/quickwit-oss/tantivy) | 0.26.1 | MIT |
| [tokenizers](https://github.com/huggingface/tokenizers) | 0.19.1 | Apache-2.0 |
| [tokenizers](https://github.com/huggingface/tokenizers) | 0.22.2 | Apache-2.0 |
| [tokio](https://github.com/tokio-rs/tokio) | 1.52.1 | MIT |
| [toml](https://github.com/toml-rs/toml) | 1.1.2+spec-1.1.0 | MIT OR Apache-2.0 |
| [tracing](https://github.com/tokio-rs/tracing) | 0.1.44 | MIT |
| [tracing-subscriber](https://github.com/tokio-rs/tracing) | 0.3.23 | MIT |
| [urlencoding](https://github.com/kornelski/rust_urlencoding) | 2.1.3 | MIT |
| [wgpu](https://github.com/gfx-rs/wgpu) | 0.19.4 | MIT OR Apache-2.0 |

Notable: the model runtime is built on **candle** (candle-core / candle-nn / candle-transformers,
Apache-2.0, Hugging Face) with **cudarc** (MIT OR Apache-2.0) for CUDA, and **tokenizers**
(Apache-2.0, Hugging Face) for tokenization.

## Full dependency graph (615 crates)

| crate | version | license |
|---|---|---|
| [adler2](https://github.com/oyvindln/adler2) | 2.0.1 | 0BSD OR MIT OR Apache-2.0 |
| [ahash](https://github.com/tkaitchuck/ahash) | 0.7.8 | MIT OR Apache-2.0 |
| [ahash](https://github.com/tkaitchuck/ahash) | 0.8.12 | MIT OR Apache-2.0 |
| [aho-corasick](https://github.com/BurntSushi/aho-corasick) | 1.1.4 | Unlicense OR MIT |
| [allocator-api2](https://github.com/zakarumych/allocator-api2) | 0.2.21 | MIT OR Apache-2.0 |
| [android_system_properties](https://github.com/nical/android_system_properties) | 0.1.5 | MIT/Apache-2.0 |
| [anstream](https://github.com/rust-cli/anstyle.git) | 1.0.0 | MIT OR Apache-2.0 |
| [anstyle](https://github.com/rust-cli/anstyle.git) | 1.0.14 | MIT OR Apache-2.0 |
| [anstyle-parse](https://github.com/rust-cli/anstyle.git) | 1.0.0 | MIT OR Apache-2.0 |
| [anstyle-query](https://github.com/rust-cli/anstyle.git) | 1.1.5 | MIT OR Apache-2.0 |
| [anstyle-wincon](https://github.com/rust-cli/anstyle.git) | 3.0.11 | MIT OR Apache-2.0 |
| [anyhow](https://github.com/dtolnay/anyhow) | 1.0.102 | MIT OR Apache-2.0 |
| [approx](https://github.com/brendanzab/approx) | 0.5.1 | Apache-2.0 |
| [arc-swap](https://github.com/vorner/arc-swap) | 1.9.1 | MIT OR Apache-2.0 |
| [arrayvec](https://github.com/bluss/arrayvec) | 0.7.6 | MIT OR Apache-2.0 |
| [ash](https://github.com/MaikKlein/ash) | 0.37.3+1.3.251 | MIT OR Apache-2.0 |
| [async-trait](https://github.com/dtolnay/async-trait) | 0.1.89 | MIT OR Apache-2.0 |
| [atomic-waker](https://github.com/smol-rs/atomic-waker) | 1.1.2 | Apache-2.0 OR MIT |
| [autocfg](https://github.com/cuviper/autocfg) | 1.5.0 | Apache-2.0 OR MIT |
| [axum](https://github.com/tokio-rs/axum) | 0.7.9 | MIT |
| [axum-core](https://github.com/tokio-rs/axum) | 0.4.5 | MIT |
| [base64](https://github.com/marshallpierce/rust-base64) | 0.13.1 | MIT/Apache-2.0 |
| [base64](https://github.com/marshallpierce/rust-base64) | 0.21.7 | MIT OR Apache-2.0 |
| [base64](https://github.com/marshallpierce/rust-base64) | 0.22.1 | MIT OR Apache-2.0 |
| [base64ct](https://github.com/RustCrypto/formats) | 1.8.3 | Apache-2.0 OR MIT |
| [bincode](https://github.com/servo/bincode) | 1.3.3 | MIT |
| [bit-set](https://github.com/contain-rs/bit-set) | 0.5.3 | MIT/Apache-2.0 |
| [bit-set](https://github.com/contain-rs/bit-set) | 0.8.0 | Apache-2.0 OR MIT |
| [bit-vec](https://github.com/contain-rs/bit-vec) | 0.6.3 | MIT/Apache-2.0 |
| [bit-vec](https://github.com/contain-rs/bit-vec) | 0.8.0 | Apache-2.0 OR MIT |
| [bitflags](https://github.com/bitflags/bitflags) | 1.3.2 | MIT/Apache-2.0 |
| [bitflags](https://github.com/bitflags/bitflags) | 2.11.1 | MIT OR Apache-2.0 |
| [bitpacking](https://github.com/quickwit-oss/bitpacking) | 0.9.3 | MIT |
| [bitvec](https://github.com/bitvecto-rs/bitvec) | 1.0.1 | MIT |
| [block](http://github.com/SSheldon/rust-block) | 0.1.6 | MIT |
| [block-buffer](https://github.com/RustCrypto/utils) | 0.10.4 | MIT OR Apache-2.0 |
| [bon](https://github.com/elastio/bon) | 3.9.1 | MIT OR Apache-2.0 |
| [bon-macros](https://github.com/elastio/bon) | 3.9.1 | MIT OR Apache-2.0 |
| [bumpalo](https://github.com/fitzgen/bumpalo) | 3.20.2 | MIT OR Apache-2.0 |
| [bytecheck](https://github.com/djkoloski/bytecheck) | 0.6.12 | MIT |
| [bytecheck_derive](https://github.com/djkoloski/bytecheck) | 0.6.12 | MIT |
| [bytemuck](https://github.com/Lokathor/bytemuck) | 1.25.0 | Zlib OR Apache-2.0 OR MIT |
| [bytemuck_derive](https://github.com/Lokathor/bytemuck) | 1.10.2 | Zlib OR Apache-2.0 OR MIT |
| [byteorder](https://github.com/BurntSushi/byteorder) | 1.5.0 | Unlicense OR MIT |
| [bytes](https://github.com/tokio-rs/bytes) | 1.11.1 | MIT |
| [candle-core](https://github.com/huggingface/candle) | 0.10.2 | MIT OR Apache-2.0 |
| [candle-kernels](https://github.com/huggingface/candle) | 0.10.2 | MIT OR Apache-2.0 |
| [candle-nn](https://github.com/huggingface/candle) | 0.10.2 | MIT OR Apache-2.0 |
| [candle-transformers](https://github.com/huggingface/candle) | 0.10.2 | MIT OR Apache-2.0 |
| [candle-ug](https://github.com/huggingface/candle) | 0.10.2 | MIT OR Apache-2.0 |
| [castaway](https://github.com/sagebind/castaway) | 0.2.4 | MIT |
| [cc](https://github.com/rust-lang/cc-rs) | 1.2.61 | MIT OR Apache-2.0 |
| [census](https://github.com/quickwit-inc/census) | 0.4.2 | MIT |
| [cfg-if](https://github.com/rust-lang/cfg-if) | 1.0.4 | MIT OR Apache-2.0 |
| [cfg_aliases](https://github.com/katharostech/cfg_aliases) | 0.1.1 | MIT |
| [chrono](https://github.com/chronotope/chrono) | 0.4.44 | MIT OR Apache-2.0 |
| [clap](https://github.com/clap-rs/clap) | 4.6.1 | MIT OR Apache-2.0 |
| [clap_builder](https://github.com/clap-rs/clap) | 4.6.0 | MIT OR Apache-2.0 |
| [clap_derive](https://github.com/clap-rs/clap) | 4.6.1 | MIT OR Apache-2.0 |
| [clap_lex](https://github.com/clap-rs/clap) | 1.1.0 | MIT OR Apache-2.0 |
| [codespan-reporting](https://github.com/brendanzab/codespan) | 0.11.1 | Apache-2.0 |
| [colorchoice](https://github.com/rust-cli/anstyle.git) | 1.0.5 | MIT OR Apache-2.0 |
| [com](https://github.com/microsoft/com-rs) | 0.6.0 | MIT |
| [com_macros](https://github.com/microsoft/com-rs) | 0.6.0 | MIT |
| [com_macros_support](https://github.com/microsoft/com-rs) | 0.6.0 | MIT |
| [combine](https://github.com/Marwes/combine) | 4.6.7 | MIT |
| [compact_str](https://github.com/ParkMyCar/compact_str) | 0.9.0 | MIT |
| [console](https://github.com/console-rs/console) | 0.15.11 | MIT |
| [const-oid](https://github.com/RustCrypto/formats/tree/master/const-oid) | 0.9.6 | Apache-2.0 OR MIT |
| [core-foundation](https://github.com/servo/core-foundation-rs) | 0.10.1 | MIT OR Apache-2.0 |
| [core-foundation](https://github.com/servo/core-foundation-rs) | 0.9.4 | MIT OR Apache-2.0 |
| [core-foundation-sys](https://github.com/servo/core-foundation-rs) | 0.8.7 | MIT OR Apache-2.0 |
| [core-graphics-types](https://github.com/servo/core-foundation-rs) | 0.1.3 | MIT OR Apache-2.0 |
| cpu-time | 1.0.0 | MIT/Apache-2.0 |
| [cpufeatures](https://github.com/RustCrypto/utils) | 0.2.17 | MIT OR Apache-2.0 |
| [crc32fast](https://github.com/srijs/rust-crc32fast) | 1.5.0 | MIT OR Apache-2.0 |
| [crossbeam-channel](https://github.com/crossbeam-rs/crossbeam) | 0.5.15 | MIT OR Apache-2.0 |
| [crossbeam-deque](https://github.com/crossbeam-rs/crossbeam) | 0.8.6 | MIT OR Apache-2.0 |
| [crossbeam-epoch](https://github.com/crossbeam-rs/crossbeam) | 0.9.18 | MIT OR Apache-2.0 |
| [crossbeam-utils](https://github.com/crossbeam-rs/crossbeam) | 0.8.21 | MIT OR Apache-2.0 |
| [crunchy](https://github.com/eira-fransham/crunchy) | 0.2.4 | MIT |
| [crypto-common](https://github.com/RustCrypto/traits) | 0.1.7 | MIT OR Apache-2.0 |
| [cudaforge](https://github.com/guoqingbao/cudaforge) | 0.1.6 | MIT OR Apache-2.0 |
| [cudarc](https://github.com/coreylowman/cudarc) | 0.17.8 | MIT OR Apache-2.0 |
| [cudarc](https://github.com/chelsea0x3b/cudarc) | 0.19.8 | MIT OR Apache-2.0 |
| [curve25519-dalek](https://github.com/dalek-cryptography/curve25519-dalek/tree/main/curve25519-dalek) | 4.1.3 | BSD-3-Clause |
| [curve25519-dalek-derive](https://github.com/dalek-cryptography/curve25519-dalek) | 0.1.1 | MIT/Apache-2.0 |
| [d3d12](https://github.com/gfx-rs/wgpu/tree/trunk/d3d12) | 0.19.0 | MIT OR Apache-2.0 |
| [darling](https://github.com/TedDriggs/darling) | 0.20.11 | MIT |
| [darling](https://github.com/TedDriggs/darling) | 0.23.0 | MIT |
| [darling_core](https://github.com/TedDriggs/darling) | 0.20.11 | MIT |
| [darling_core](https://github.com/TedDriggs/darling) | 0.23.0 | MIT |
| [darling_macro](https://github.com/TedDriggs/darling) | 0.20.11 | MIT |
| [darling_macro](https://github.com/TedDriggs/darling) | 0.23.0 | MIT |
| [dary_heap](https://github.com/hanmertens/dary_heap) | 0.3.9 | MIT OR Apache-2.0 |
| [data-encoding](https://github.com/ia0/data-encoding) | 2.11.0 | MIT |
| [datasketches](https://github.com/apache/datasketches-rust) | 0.2.0 | Apache-2.0 |
| [der](https://github.com/RustCrypto/formats/tree/master/der) | 0.7.10 | Apache-2.0 OR MIT |
| [deranged](https://github.com/jhpratt/deranged) | 0.5.8 | MIT OR Apache-2.0 |
| [derive_builder](https://github.com/colin-kiegel/rust-derive-builder) | 0.20.2 | MIT OR Apache-2.0 |
| [derive_builder_core](https://github.com/colin-kiegel/rust-derive-builder) | 0.20.2 | MIT OR Apache-2.0 |
| [derive_builder_macro](https://github.com/colin-kiegel/rust-derive-builder) | 0.20.2 | MIT OR Apache-2.0 |
| [digest](https://github.com/RustCrypto/traits) | 0.10.7 | MIT OR Apache-2.0 |
| [dirs](https://github.com/soc/dirs-rs) | 5.0.1 | MIT OR Apache-2.0 |
| [dirs-sys](https://github.com/dirs-dev/dirs-sys-rs) | 0.4.1 | MIT OR Apache-2.0 |
| [displaydoc](https://github.com/yaahc/displaydoc) | 0.2.5 | MIT OR Apache-2.0 |
| [downcast-rs](https://github.com/marcianx/downcast-rs) | 2.0.2 | MIT OR Apache-2.0 |
| [dyn-stack](https://codeberg.org/sarah-quinones/dyn-stack) | 0.13.2 | MIT |
| [dyn-stack-macros](https://github.com/kitegi/dynstack/) | 0.1.3 | MIT |
| [ed25519](https://github.com/RustCrypto/signatures/tree/master/ed25519) | 2.2.3 | Apache-2.0 OR MIT |
| [ed25519-dalek](https://github.com/dalek-cryptography/curve25519-dalek/tree/main/ed25519-dalek) | 2.2.0 | BSD-3-Clause |
| [either](https://github.com/rayon-rs/either) | 1.15.0 | MIT OR Apache-2.0 |
| [encode_unicode](https://github.com/tormol/encode_unicode) | 1.0.0 | Apache-2.0 OR MIT |
| [encoding_rs](https://github.com/hsivonen/encoding_rs) | 0.8.35 | (Apache-2.0 OR MIT) AND BSD-3-Clause |
| [enum-as-inner](https://github.com/bluejekyll/enum-as-inner) | 0.6.1 | MIT/Apache-2.0 |
| [env_home](https://github.com/notpeter/env-home) | 0.1.0 | MIT OR Apache-2.0 |
| [env_logger](https://github.com/rust-cli/env_logger) | 0.10.2 | MIT OR Apache-2.0 |
| [equivalent](https://github.com/indexmap-rs/equivalent) | 1.0.2 | Apache-2.0 OR MIT |
| [erased-serde](https://github.com/dtolnay/erased-serde) | 0.4.10 | MIT OR Apache-2.0 |
| [errno](https://github.com/lambda-fairy/rust-errno) | 0.3.14 | MIT OR Apache-2.0 |
| [esaxx-rs](https://github.com/Narsil/esaxx-rs) | 0.1.10 | Apache-2.0 |
| [fallible-iterator](https://github.com/sfackler/rust-fallible-iterator) | 0.3.0 | MIT/Apache-2.0 |
| [fallible-streaming-iterator](https://github.com/sfackler/fallible-streaming-iterator) | 0.1.9 | MIT/Apache-2.0 |
| [fancy-regex](https://github.com/fancy-regex/fancy-regex) | 0.17.0 | MIT |
| [fastdivide](https://github.com/fulmicoton/fastdivide) | 0.4.2 | zlib-acknowledgement OR MIT |
| [fastrand](https://github.com/smol-rs/fastrand) | 2.4.1 | Apache-2.0 OR MIT |
| [fiat-crypto](https://github.com/mit-plv/fiat-crypto) | 0.2.9 | MIT OR Apache-2.0 OR BSD-1-Clause |
| [filetime](https://github.com/alexcrichton/filetime) | 0.2.27 | MIT/Apache-2.0 |
| [find-msvc-tools](https://github.com/rust-lang/cc-rs) | 0.1.9 | MIT OR Apache-2.0 |
| [flate2](https://github.com/rust-lang/flate2-rs) | 1.1.9 | MIT OR Apache-2.0 |
| [float8](https://github.com/EricLBuehler/float8) | 0.7.0 | MIT |
| [fnv](https://github.com/servo/rust-fnv) | 1.0.7 | Apache-2.0 / MIT |
| [foldhash](https://github.com/orlp/foldhash) | 0.1.5 | Zlib |
| [foldhash](https://github.com/orlp/foldhash) | 0.2.0 | Zlib |
| [foreign-types](https://github.com/sfackler/foreign-types) | 0.3.2 | MIT/Apache-2.0 |
| [foreign-types](https://github.com/sfackler/foreign-types) | 0.5.0 | MIT/Apache-2.0 |
| [foreign-types-macros](https://github.com/sfackler/foreign-types) | 0.2.3 | MIT/Apache-2.0 |
| [foreign-types-shared](https://github.com/sfackler/foreign-types) | 0.1.1 | MIT/Apache-2.0 |
| [foreign-types-shared](https://github.com/sfackler/foreign-types) | 0.3.1 | MIT/Apache-2.0 |
| [form_urlencoded](https://github.com/servo/rust-url) | 1.2.2 | MIT OR Apache-2.0 |
| [friedrich](https://github.com/nestordemeure/friedrich) | 0.5.0 | Apache-2.0 |
| [fs2](https://github.com/danburkert/fs2-rs) | 0.4.3 | MIT/Apache-2.0 |
| [fs4](https://github.com/al8n/fs4-rs) | 0.13.1 | MIT OR Apache-2.0 |
| [fsevent-sys](https://github.com/octplane/fsevent-rust/tree/master/fsevent-sys) | 4.1.0 | MIT |
| [funty](https://github.com/myrrlyn/funty) | 2.0.0 | MIT |
| [futures](https://github.com/rust-lang/futures-rs) | 0.3.32 | MIT OR Apache-2.0 |
| [futures-channel](https://github.com/rust-lang/futures-rs) | 0.3.32 | MIT OR Apache-2.0 |
| [futures-core](https://github.com/rust-lang/futures-rs) | 0.3.32 | MIT OR Apache-2.0 |
| [futures-executor](https://github.com/rust-lang/futures-rs) | 0.3.32 | MIT OR Apache-2.0 |
| [futures-io](https://github.com/rust-lang/futures-rs) | 0.3.32 | MIT OR Apache-2.0 |
| [futures-macro](https://github.com/rust-lang/futures-rs) | 0.3.32 | MIT OR Apache-2.0 |
| [futures-sink](https://github.com/rust-lang/futures-rs) | 0.3.32 | MIT OR Apache-2.0 |
| [futures-task](https://github.com/rust-lang/futures-rs) | 0.3.32 | MIT OR Apache-2.0 |
| [futures-util](https://github.com/rust-lang/futures-rs) | 0.3.32 | MIT OR Apache-2.0 |
| [gemm](https://github.com/sarah-ek/gemm/) | 0.18.2 | MIT |
| [gemm](https://github.com/sarah-ek/gemm/) | 0.19.0 | MIT |
| [gemm-c32](https://github.com/sarah-ek/gemm/) | 0.18.2 | MIT |
| [gemm-c32](https://github.com/sarah-ek/gemm/) | 0.19.0 | MIT |
| [gemm-c64](https://github.com/sarah-ek/gemm/) | 0.18.2 | MIT |
| [gemm-c64](https://github.com/sarah-ek/gemm/) | 0.19.0 | MIT |
| [gemm-common](https://github.com/sarah-ek/gemm/) | 0.18.2 | MIT |
| [gemm-common](https://github.com/sarah-ek/gemm/) | 0.19.0 | MIT |
| [gemm-f16](https://github.com/sarah-ek/gemm/) | 0.18.2 | MIT |
| [gemm-f16](https://github.com/sarah-ek/gemm/) | 0.19.0 | MIT |
| [gemm-f32](https://github.com/sarah-ek/gemm/) | 0.18.2 | MIT |
| [gemm-f32](https://github.com/sarah-ek/gemm/) | 0.19.0 | MIT |
| [gemm-f64](https://github.com/sarah-ek/gemm/) | 0.18.2 | MIT |
| [gemm-f64](https://github.com/sarah-ek/gemm/) | 0.19.0 | MIT |
| [generic-array](https://github.com/fizyk20/generic-array.git) | 0.14.7 | MIT |
| [getrandom](https://github.com/rust-random/getrandom) | 0.2.17 | MIT OR Apache-2.0 |
| [getrandom](https://github.com/rust-random/getrandom) | 0.3.4 | MIT OR Apache-2.0 |
| [getrandom](https://github.com/rust-random/getrandom) | 0.4.2 | MIT OR Apache-2.0 |
| [gl_generator](https://github.com/brendanzab/gl-rs/) | 0.14.0 | Apache-2.0 |
| [glam](https://github.com/bitshifter/glam-rs) | 0.27.0 | MIT OR Apache-2.0 |
| [glob](https://github.com/rust-lang/glob) | 0.3.3 | MIT OR Apache-2.0 |
| [glow](https://github.com/grovesNL/glow) | 0.13.1 | MIT OR Apache-2.0 OR Zlib |
| [glutin_wgl_sys](https://github.com/rust-windowing/glutin) | 0.5.0 | Apache-2.0 |
| [gpu-alloc](https://github.com/zakarumych/gpu-alloc) | 0.6.0 | MIT OR Apache-2.0 |
| [gpu-alloc-types](https://github.com/zakarumych/gpu-alloc) | 0.3.0 | MIT OR Apache-2.0 |
| [gpu-allocator](https://github.com/Traverse-Research/gpu-allocator) | 0.25.0 | MIT OR Apache-2.0 |
| [gpu-descriptor](https://github.com/zakarumych/gpu-descriptor) | 0.2.4 | MIT OR Apache-2.0 |
| [gpu-descriptor-types](https://github.com/zakarumych/gpu-descriptor) | 0.1.2 | MIT OR Apache-2.0 |
| [h2](https://github.com/hyperium/h2) | 0.3.27 | MIT |
| [half](https://github.com/VoidStarKat/half-rs) | 2.7.1 | MIT OR Apache-2.0 |
| [hashbrown](https://github.com/rust-lang/hashbrown) | 0.12.3 | MIT OR Apache-2.0 |
| [hashbrown](https://github.com/rust-lang/hashbrown) | 0.14.5 | MIT OR Apache-2.0 |
| [hashbrown](https://github.com/rust-lang/hashbrown) | 0.15.5 | MIT OR Apache-2.0 |
| [hashbrown](https://github.com/rust-lang/hashbrown) | 0.16.1 | MIT OR Apache-2.0 |
| [hashbrown](https://github.com/rust-lang/hashbrown) | 0.17.0 | MIT OR Apache-2.0 |
| [hashlink](https://github.com/kyren/hashlink) | 0.8.4 | MIT OR Apache-2.0 |
| [hassle-rs](https://github.com/Traverse-Research/hassle-rs) | 0.11.0 | MIT |
| [heck](https://github.com/withoutboats/heck) | 0.5.0 | MIT OR Apache-2.0 |
| [hermit-abi](https://github.com/hermit-os/hermit-rs) | 0.5.2 | MIT OR Apache-2.0 |
| [hexf-parse](https://github.com/lifthrasiir/hexf) | 0.2.1 | CC0-1.0 |
| [hf-hub](https://github.com/huggingface/hf-hub) | 0.3.2 | Apache-2.0 |
| [hnsw_rs](https://github.com/jean-pierreBoth/hnswlib-rs) | 0.2.1 | MIT/Apache-2.0 |
| [htmlescape](https://github.com/veddan/rust-htmlescape) | 0.3.1 | Apache-2.0 / MIT / MPL-2.0 |
| [http](https://github.com/hyperium/http) | 0.2.12 | MIT OR Apache-2.0 |
| [http](https://github.com/hyperium/http) | 1.4.0 | MIT OR Apache-2.0 |
| [http-body](https://github.com/hyperium/http-body) | 0.4.6 | MIT |
| [http-body](https://github.com/hyperium/http-body) | 1.0.1 | MIT |
| [http-body-util](https://github.com/hyperium/http-body) | 0.1.3 | MIT |
| [httparse](https://github.com/seanmonstar/httparse) | 1.10.1 | MIT OR Apache-2.0 |
| [httpdate](https://github.com/pyfisch/httpdate) | 1.0.3 | MIT OR Apache-2.0 |
| [humantime](https://github.com/chronotope/humantime) | 2.3.0 | MIT OR Apache-2.0 |
| [hyper](https://github.com/hyperium/hyper) | 0.14.32 | MIT |
| [hyper](https://github.com/hyperium/hyper) | 1.9.0 | MIT |
| [hyper-tls](https://github.com/hyperium/hyper-tls) | 0.5.0 | MIT/Apache-2.0 |
| [hyper-util](https://github.com/hyperium/hyper-util) | 0.1.20 | MIT |
| [iana-time-zone](https://github.com/strawlab/iana-time-zone) | 0.1.65 | MIT OR Apache-2.0 |
| [iana-time-zone-haiku](https://github.com/strawlab/iana-time-zone) | 0.1.2 | MIT OR Apache-2.0 |
| [icu_collections](https://github.com/unicode-org/icu4x) | 2.2.0 | Unicode-3.0 |
| [icu_locale_core](https://github.com/unicode-org/icu4x) | 2.2.0 | Unicode-3.0 |
| [icu_normalizer](https://github.com/unicode-org/icu4x) | 2.2.0 | Unicode-3.0 |
| [icu_normalizer_data](https://github.com/unicode-org/icu4x) | 2.2.0 | Unicode-3.0 |
| [icu_properties](https://github.com/unicode-org/icu4x) | 2.2.0 | Unicode-3.0 |
| [icu_properties_data](https://github.com/unicode-org/icu4x) | 2.2.0 | Unicode-3.0 |
| [icu_provider](https://github.com/unicode-org/icu4x) | 2.2.0 | Unicode-3.0 |
| [id-arena](https://github.com/fitzgen/id-arena) | 2.3.0 | MIT/Apache-2.0 |
| [ident_case](https://github.com/TedDriggs/ident_case) | 1.0.1 | MIT/Apache-2.0 |
| [idna](https://github.com/servo/rust-url/) | 1.1.0 | MIT OR Apache-2.0 |
| [idna_adapter](https://github.com/hsivonen/idna_adapter) | 1.2.1 | Apache-2.0 OR MIT |
| [indexmap](https://github.com/indexmap-rs/indexmap) | 2.14.0 | Apache-2.0 OR MIT |
| [indicatif](https://github.com/console-rs/indicatif) | 0.17.11 | MIT |
| [inotify](https://github.com/hannobraun/inotify) | 0.9.6 | ISC |
| [inotify-sys](https://github.com/hannobraun/inotify-sys) | 0.1.5 | ISC |
| [inventory](https://github.com/dtolnay/inventory) | 0.3.24 | MIT OR Apache-2.0 |
| [ipnet](https://github.com/krisprice/ipnet) | 2.12.0 | MIT OR Apache-2.0 |
| [is-terminal](https://github.com/sunfishcode/is-terminal) | 0.4.17 | MIT |
| [is_terminal_polyfill](https://github.com/polyfill-rs/is_terminal_polyfill) | 1.70.2 | MIT OR Apache-2.0 |
| [itertools](https://github.com/rust-itertools/itertools) | 0.11.0 | MIT OR Apache-2.0 |
| [itertools](https://github.com/rust-itertools/itertools) | 0.12.1 | MIT OR Apache-2.0 |
| [itertools](https://github.com/rust-itertools/itertools) | 0.14.0 | MIT OR Apache-2.0 |
| [itoa](https://github.com/dtolnay/itoa) | 1.0.18 | MIT OR Apache-2.0 |
| [jni-sys](https://github.com/jni-rs/jni-sys) | 0.3.1 | MIT OR Apache-2.0 |
| [jni-sys](https://github.com/jni-rs/jni-sys) | 0.4.1 | MIT OR Apache-2.0 |
| [jni-sys-macros](https://github.com/jni-rs/jni-sys) | 0.4.1 | MIT OR Apache-2.0 |
| [jobserver](https://github.com/rust-lang/jobserver-rs) | 0.1.34 | MIT OR Apache-2.0 |
| [js-sys](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/js-sys) | 0.3.95 | MIT OR Apache-2.0 |
| [khronos-egl](https://github.com/timothee-haudebourg/khronos-egl) | 6.0.0 | MIT/Apache-2.0 |
| [khronos_api](https://github.com/brendanzab/gl-rs/) | 3.1.0 | Apache-2.0 |
| [kqueue](https://gitlab.com/rust-kqueue/rust-kqueue) | 1.1.1 | MIT |
| [kqueue-sys](https://gitlab.com/rust-kqueue/rust-kqueue-sys) | 1.0.4 | MIT |
| [lazy_static](https://github.com/rust-lang-nursery/lazy-static.rs) | 1.5.0 | MIT OR Apache-2.0 |
| [leb128fmt](https://github.com/bluk/leb128fmt) | 0.1.0 | MIT OR Apache-2.0 |
| [levenshtein_automata](https://github.com/tantivy-search/levenshtein-automata) | 0.2.1 | MIT |
| [libc](https://github.com/rust-lang/libc) | 0.2.186 | MIT OR Apache-2.0 |
| [libloading](https://github.com/nagisa/rust_libloading/) | 0.7.4 | ISC |
| [libloading](https://github.com/nagisa/rust_libloading/) | 0.8.9 | ISC |
| [libloading](https://github.com/nagisa/rust_libloading/) | 0.9.0 | ISC |
| [libm](https://github.com/rust-lang/compiler-builtins) | 0.2.16 | MIT |
| [libredox](https://gitlab.redox-os.org/redox-os/libredox.git) | 0.1.16 | MIT |
| [libsqlite3-sys](https://github.com/rusqlite/rusqlite) | 0.27.0 | MIT |
| [linux-raw-sys](https://github.com/sunfishcode/linux-raw-sys) | 0.12.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [litemap](https://github.com/unicode-org/icu4x) | 0.8.2 | Unicode-3.0 |
| [lock_api](https://github.com/Amanieu/parking_lot) | 0.4.14 | MIT OR Apache-2.0 |
| [log](https://github.com/rust-lang/log) | 0.4.29 | MIT OR Apache-2.0 |
| [lru](https://github.com/jeromefroe/lru-rs.git) | 0.16.4 | MIT |
| [lz4_flex](https://github.com/pseitz/lz4_flex) | 0.13.0 | MIT |
| [mach2](https://github.com/JohnTitor/mach2) | 0.4.3 | BSD-2-Clause OR MIT OR Apache-2.0 |
| [macro_rules_attribute](https://github.com/danielhenrymantilla/macro_rules_attribute-rs) | 0.2.2 | Apache-2.0 OR MIT OR Zlib |
| [macro_rules_attribute-proc_macro](https://github.com/danielhenrymantilla/macro_rules_attribute-rs) | 0.2.2 | Apache-2.0 OR MIT OR Zlib |
| [malloc_buf](https://github.com/SSheldon/malloc_buf) | 0.0.6 | MIT |
| [matchit](https://github.com/ibraheemdev/matchit) | 0.7.3 | MIT AND BSD-3-Clause |
| [matrixmultiply](https://github.com/bluss/matrixmultiply/) | 0.3.10 | MIT/Apache-2.0 |
| [measure_time](https://github.com/PSeitz/rust_measure_time) | 0.9.0 | MIT |
| [memchr](https://github.com/BurntSushi/memchr) | 2.8.0 | Unlicense OR MIT |
| [memmap2](https://github.com/RazrFalcon/memmap2-rs) | 0.9.11 | MIT OR Apache-2.0 |
| [memoffset](https://github.com/Gilnaa/memoffset) | 0.6.5 | MIT |
| [memoffset](https://github.com/Gilnaa/memoffset) | 0.7.1 | MIT |
| [metal](https://github.com/gfx-rs/metal-rs) | 0.27.0 | MIT OR Apache-2.0 |
| [mime](https://github.com/hyperium/mime) | 0.3.17 | MIT OR Apache-2.0 |
| [minimal-lexical](https://github.com/Alexhuszagh/minimal-lexical) | 0.2.1 | MIT/Apache-2.0 |
| [miniz_oxide](https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide) | 0.8.9 | MIT OR Zlib OR Apache-2.0 |
| [mio](https://github.com/tokio-rs/mio) | 0.8.11 | MIT |
| [mio](https://github.com/tokio-rs/mio) | 1.2.0 | MIT |
| [mmap-rs](https://github.com/StephanvanSchaik/mmap-rs) | 0.6.1 | Apache-2.0 OR MIT |
| [monostate](https://github.com/dtolnay/monostate) | 0.1.18 | MIT OR Apache-2.0 |
| [monostate-impl](https://github.com/dtolnay/monostate) | 0.1.18 | MIT OR Apache-2.0 |
| [murmurhash32](https://github.com/quickwit-inc/murmurhash32) | 0.3.1 | MIT |
| [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) | 0.19.2 | MIT OR Apache-2.0 |
| [nalgebra](https://github.com/dimforge/nalgebra) | 0.29.0 | BSD-3-Clause |
| [nalgebra](https://github.com/dimforge/nalgebra) | 0.30.1 | BSD-3-Clause |
| [nalgebra](https://github.com/dimforge/nalgebra) | 0.32.6 | BSD-3-Clause |
| [nalgebra-macros](https://github.com/dimforge/nalgebra) | 0.1.0 | Apache-2.0 |
| [nalgebra-macros](https://github.com/dimforge/nalgebra) | 0.2.2 | Apache-2.0 |
| [native-tls](https://github.com/rust-native-tls/rust-native-tls) | 0.2.18 | MIT OR Apache-2.0 |
| [ndarray](https://github.com/rust-ndarray/ndarray) | 0.15.6 | MIT OR Apache-2.0 |
| [ndarray-npy](https://github.com/jturner314/ndarray-npy) | 0.8.1 | MIT OR Apache-2.0 |
| [ndk-sys](https://github.com/rust-mobile/ndk) | 0.5.0+25.2.9519653 | MIT OR Apache-2.0 |
| [nix](https://github.com/nix-rust/nix) | 0.23.2 | MIT |
| [nix](https://github.com/nix-rust/nix) | 0.26.4 | MIT |
| [nom](https://github.com/Geal/nom) | 7.1.3 | MIT |
| [notify](https://github.com/notify-rs/notify.git) | 6.1.1 | CC0-1.0 |
| [nu-ansi-term](https://github.com/nushell/nu-ansi-term) | 0.50.3 | MIT |
| [num](https://github.com/rust-num/num) | 0.4.3 | MIT OR Apache-2.0 |
| [num-bigint](https://github.com/rust-num/num-bigint) | 0.4.6 | MIT OR Apache-2.0 |
| [num-complex](https://github.com/rust-num/num-complex) | 0.4.6 | MIT OR Apache-2.0 |
| [num-conv](https://github.com/jhpratt/num-conv) | 0.2.1 | MIT OR Apache-2.0 |
| [num-integer](https://github.com/rust-num/num-integer) | 0.1.46 | MIT OR Apache-2.0 |
| [num-iter](https://github.com/rust-num/num-iter) | 0.1.45 | MIT OR Apache-2.0 |
| [num-rational](https://github.com/rust-num/num-rational) | 0.4.2 | MIT OR Apache-2.0 |
| [num-traits](https://github.com/rust-num/num-traits) | 0.2.19 | MIT OR Apache-2.0 |
| [num_cpus](https://github.com/seanmonstar/num_cpus) | 1.17.0 | MIT OR Apache-2.0 |
| [number_prefix](https://github.com/ogham/rust-number-prefix) | 0.4.0 | MIT |
| [objc](http://github.com/SSheldon/rust-objc) | 0.2.7 | MIT |
| [objc_exception](http://github.com/SSheldon/rust-objc-exception) | 0.1.2 | MIT |
| [once_cell](https://github.com/matklad/once_cell) | 1.21.4 | MIT OR Apache-2.0 |
| [once_cell_polyfill](https://github.com/polyfill-rs/once_cell_polyfill) | 1.70.2 | MIT OR Apache-2.0 |
| [oneshot](https://github.com/faern/oneshot) | 0.1.13 | MIT OR Apache-2.0 |
| [onig](https://github.com/iwillspeak/rust-onig) | 6.5.2 | MIT |
| [onig_sys](https://github.com/rust-onig/rust-onig) | 69.9.2 | MIT |
| [openssl](https://github.com/rust-openssl/rust-openssl) | 0.10.78 | Apache-2.0 |
| openssl-macros | 0.1.1 | MIT/Apache-2.0 |
| [openssl-probe](https://github.com/rustls/openssl-probe) | 0.2.1 | MIT OR Apache-2.0 |
| [openssl-src](https://github.com/alexcrichton/openssl-src-rs) | 300.6.0+3.6.2 | MIT/Apache-2.0 |
| [openssl-sys](https://github.com/rust-openssl/rust-openssl) | 0.9.114 | MIT |
| [option-ext](https://github.com/soc/option-ext.git) | 0.2.0 | MPL-2.0 |
| [ordered-float](https://github.com/reem/rust-ordered-float) | 5.3.0 | MIT |
| [ownedbytes](https://github.com/quickwit-oss/tantivy) | 0.9.0 | MIT |
| [parking_lot](https://github.com/Amanieu/parking_lot) | 0.12.5 | MIT OR Apache-2.0 |
| [parking_lot_core](https://github.com/Amanieu/parking_lot) | 0.9.12 | MIT OR Apache-2.0 |
| [paste](https://github.com/dtolnay/paste) | 1.0.15 | MIT OR Apache-2.0 |
| [percent-encoding](https://github.com/servo/rust-url/) | 2.3.2 | MIT OR Apache-2.0 |
| [pest](https://github.com/pest-parser/pest) | 2.8.6 | MIT OR Apache-2.0 |
| [pest_derive](https://github.com/pest-parser/pest) | 2.8.6 | MIT OR Apache-2.0 |
| [pest_generator](https://github.com/pest-parser/pest) | 2.8.6 | MIT OR Apache-2.0 |
| [pest_meta](https://github.com/pest-parser/pest) | 2.8.6 | MIT OR Apache-2.0 |
| [pin-project-lite](https://github.com/taiki-e/pin-project-lite) | 0.2.17 | Apache-2.0 OR MIT |
| [pin-utils](https://github.com/rust-lang-nursery/pin-utils) | 0.1.0 | MIT OR Apache-2.0 |
| [pkcs8](https://github.com/RustCrypto/formats/tree/master/pkcs8) | 0.10.2 | Apache-2.0 OR MIT |
| [pkg-config](https://github.com/rust-lang/pkg-config-rs) | 0.3.33 | MIT OR Apache-2.0 |
| [plain](https://github.com/randomites/plain) | 0.2.3 | MIT/Apache-2.0 |
| [portable-atomic](https://github.com/taiki-e/portable-atomic) | 1.13.1 | Apache-2.0 OR MIT |
| [potential_utf](https://github.com/unicode-org/icu4x) | 0.1.5 | Unicode-3.0 |
| [powerfmt](https://github.com/jhpratt/powerfmt) | 0.2.0 | MIT OR Apache-2.0 |
| [ppv-lite86](https://github.com/cryptocorrosion/cryptocorrosion) | 0.2.21 | MIT OR Apache-2.0 |
| [presser](https://github.com/EmbarkStudios/presser) | 0.3.1 | MIT OR Apache-2.0 |
| [prettyplease](https://github.com/dtolnay/prettyplease) | 0.2.37 | MIT OR Apache-2.0 |
| [proc-macro2](https://github.com/dtolnay/proc-macro2) | 1.0.106 | MIT OR Apache-2.0 |
| [profiling](https://github.com/aclysma/profiling) | 1.0.17 | MIT OR Apache-2.0 |
| [ptr_meta](https://github.com/djkoloski/ptr_meta) | 0.1.4 | MIT |
| [ptr_meta_derive](https://github.com/djkoloski/ptr_meta) | 0.1.4 | MIT |
| [pulp](https://github.com/sarah-ek/pulp/) | 0.21.5 | MIT |
| [pulp](https://github.com/sarah-quinones/pulp/) | 0.22.2 | MIT |
| [pulp-wasm-simd-flag](https://github.com/sarah-quinones/pulp/) | 0.1.0 | MIT |
| [py_literal](https://github.com/jturner314/py_literal) | 0.4.0 | MIT OR Apache-2.0 |
| [quote](https://github.com/dtolnay/quote) | 1.0.45 | MIT OR Apache-2.0 |
| [r-efi](https://github.com/r-efi/r-efi) | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later |
| [r-efi](https://github.com/r-efi/r-efi) | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later |
| [radium](https://github.com/bitvecto-rs/radium) | 0.7.0 | MIT |
| [rand](https://github.com/rust-random/rand) | 0.8.6 | MIT OR Apache-2.0 |
| [rand](https://github.com/rust-random/rand) | 0.9.4 | MIT OR Apache-2.0 |
| [rand_chacha](https://github.com/rust-random/rand) | 0.3.1 | MIT OR Apache-2.0 |
| [rand_chacha](https://github.com/rust-random/rand) | 0.9.0 | MIT OR Apache-2.0 |
| [rand_core](https://github.com/rust-random/rand) | 0.6.4 | MIT OR Apache-2.0 |
| [rand_core](https://github.com/rust-random/rand) | 0.9.5 | MIT OR Apache-2.0 |
| [rand_distr](https://github.com/rust-random/rand) | 0.4.3 | MIT OR Apache-2.0 |
| [rand_distr](https://github.com/rust-random/rand_distr) | 0.5.1 | MIT OR Apache-2.0 |
| [range-alloc](https://github.com/gfx-rs/range-alloc) | 0.1.5 | MIT OR Apache-2.0 |
| [raw-cpuid](https://github.com/gz/rust-cpuid) | 11.6.0 | MIT |
| [raw-window-handle](https://github.com/rust-windowing/raw-window-handle) | 0.6.2 | MIT OR Apache-2.0 OR Zlib |
| [rawpointer](https://github.com/bluss/rawpointer/) | 0.2.1 | MIT/Apache-2.0 |
| [rayon](https://github.com/rayon-rs/rayon) | 1.12.0 | MIT OR Apache-2.0 |
| [rayon-cond](https://github.com/cuviper/rayon-cond) | 0.3.0 | Apache-2.0/MIT |
| [rayon-cond](https://github.com/cuviper/rayon-cond) | 0.4.0 | Apache-2.0/MIT |
| [rayon-core](https://github.com/rayon-rs/rayon) | 1.13.0 | MIT OR Apache-2.0 |
| [reborrow](https://github.com/sarah-ek/reborrow/) | 0.5.5 | MIT |
| [redox_syscall](https://gitlab.redox-os.org/redox-os/syscall) | 0.5.18 | MIT |
| [redox_syscall](https://gitlab.redox-os.org/redox-os/syscall) | 0.7.4 | MIT |
| [redox_users](https://gitlab.redox-os.org/redox-os/users) | 0.4.6 | MIT |
| [regex](https://github.com/rust-lang/regex) | 1.12.3 | MIT OR Apache-2.0 |
| [regex-automata](https://github.com/rust-lang/regex) | 0.4.14 | MIT OR Apache-2.0 |
| [regex-syntax](https://github.com/rust-lang/regex) | 0.8.10 | MIT OR Apache-2.0 |
| [rend](https://github.com/djkoloski/rend) | 0.4.2 | MIT |
| [renderdoc-sys](https://github.com/ebkalderon/renderdoc-rs) | 1.1.0 | MIT OR Apache-2.0 |
| [reqwest](https://github.com/seanmonstar/reqwest) | 0.11.27 | MIT OR Apache-2.0 |
| [ring](https://github.com/briansmith/ring) | 0.17.14 | Apache-2.0 AND ISC |
| [rkyv](https://github.com/rkyv/rkyv) | 0.7.46 | MIT |
| [rkyv_derive](https://github.com/rkyv/rkyv) | 0.7.46 | MIT |
| [rusqlite](https://github.com/rusqlite/rusqlite) | 0.30.0 | MIT |
| [rust-stemmers](https://github.com/CurrySoftware/rust-stemmers) | 1.2.0 | MIT/BSD-3-Clause |
| [rustc-hash](https://github.com/rust-lang-nursery/rustc-hash) | 1.1.0 | Apache-2.0/MIT |
| [rustc-hash](https://github.com/rust-lang/rustc-hash) | 2.1.2 | Apache-2.0 OR MIT |
| [rustc_version](https://github.com/djc/rustc-version-rs) | 0.4.1 | MIT OR Apache-2.0 |
| [rustix](https://github.com/bytecodealliance/rustix) | 1.1.4 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [rustls](https://github.com/rustls/rustls) | 0.23.39 | Apache-2.0 OR ISC OR MIT |
| [rustls-pemfile](https://github.com/rustls/pemfile) | 1.0.4 | Apache-2.0 OR ISC OR MIT |
| [rustls-pki-types](https://github.com/rustls/pki-types) | 1.14.1 | MIT OR Apache-2.0 |
| [rustls-webpki](https://github.com/rustls/webpki) | 0.103.13 | ISC |
| [rustversion](https://github.com/dtolnay/rustversion) | 1.0.22 | MIT OR Apache-2.0 |
| [ryu](https://github.com/dtolnay/ryu) | 1.0.23 | Apache-2.0 OR BSL-1.0 |
| [safe_arch](https://github.com/Lokathor/safe_arch) | 0.7.4 | Zlib OR Apache-2.0 OR MIT |
| [safetensors](https://github.com/huggingface/safetensors) | 0.4.5 | Apache-2.0 |
| [safetensors](https://github.com/huggingface/safetensors) | 0.7.0 | Apache-2.0 |
| [same-file](https://github.com/BurntSushi/same-file) | 1.0.6 | Unlicense/MIT |
| [schannel](https://github.com/steffengy/schannel-rs) | 0.1.29 | MIT |
| [scopeguard](https://github.com/bluss/scopeguard) | 1.2.0 | MIT OR Apache-2.0 |
| [seahash](https://gitlab.redox-os.org/redox-os/seahash) | 4.1.0 | MIT |
| [security-framework](https://github.com/kornelski/rust-security-framework) | 3.7.0 | MIT OR Apache-2.0 |
| [security-framework-sys](https://github.com/kornelski/rust-security-framework) | 2.17.0 | MIT OR Apache-2.0 |
| [semver](https://github.com/dtolnay/semver) | 1.0.28 | MIT OR Apache-2.0 |
| [seq-macro](https://github.com/dtolnay/seq-macro) | 0.3.6 | MIT OR Apache-2.0 |
| [serde](https://github.com/serde-rs/serde) | 1.0.228 | MIT OR Apache-2.0 |
| [serde-big-array](https://github.com/est31/serde-big-array) | 0.5.1 | MIT OR Apache-2.0 |
| [serde_core](https://github.com/serde-rs/serde) | 1.0.228 | MIT OR Apache-2.0 |
| [serde_derive](https://github.com/serde-rs/serde) | 1.0.228 | MIT OR Apache-2.0 |
| [serde_json](https://github.com/serde-rs/json) | 1.0.149 | MIT OR Apache-2.0 |
| [serde_path_to_error](https://github.com/dtolnay/path-to-error) | 0.1.20 | MIT OR Apache-2.0 |
| [serde_plain](https://github.com/mitsuhiko/serde-plain) | 1.0.2 | MIT/Apache-2.0 |
| [serde_spanned](https://github.com/toml-rs/toml) | 1.1.1 | MIT OR Apache-2.0 |
| [serde_urlencoded](https://github.com/nox/serde_urlencoded) | 0.7.1 | MIT/Apache-2.0 |
| [sha1](https://github.com/RustCrypto/hashes) | 0.10.6 | MIT OR Apache-2.0 |
| [sha2](https://github.com/RustCrypto/hashes) | 0.10.9 | MIT OR Apache-2.0 |
| [sharded-slab](https://github.com/hawkw/sharded-slab) | 0.1.7 | MIT |
| [shared_memory](https://github.com/elast0ny/shared_memory-rs) | 0.12.4 | MIT OR Apache-2.0 |
| [shlex](https://github.com/comex/rust-shlex) | 1.3.0 | MIT OR Apache-2.0 |
| [signal-hook-registry](https://github.com/vorner/signal-hook) | 1.4.8 | MIT OR Apache-2.0 |
| [signature](https://github.com/RustCrypto/traits/tree/master/signature) | 2.2.0 | Apache-2.0 OR MIT |
| [simba](https://github.com/dimforge/simba) | 0.6.0 | Apache-2.0 |
| [simba](https://github.com/dimforge/simba) | 0.7.3 | Apache-2.0 |
| [simba](https://github.com/dimforge/simba) | 0.8.1 | Apache-2.0 |
| [simd-adler32](https://github.com/mcountryman/simd-adler32) | 0.3.9 | MIT |
| [simdutf8](https://github.com/rusticstuff/simdutf8) | 0.1.5 | MIT OR Apache-2.0 |
| [sketches-ddsketch](https://github.com/mheffner/rust-sketches-ddsketch) | 0.4.0 | Apache-2.0 |
| [skiplist](https://www.github.com/JP-Ellis/rust-skiplist/) | 0.5.1 | MIT |
| [slab](https://github.com/tokio-rs/slab) | 0.4.12 | MIT |
| [slotmap](https://github.com/orlp/slotmap) | 1.1.1 | Zlib |
| [smallvec](https://github.com/servo/rust-smallvec) | 1.15.1 | MIT OR Apache-2.0 |
| [socket2](https://github.com/rust-lang/socket2) | 0.5.10 | MIT OR Apache-2.0 |
| [socket2](https://github.com/rust-lang/socket2) | 0.6.3 | MIT OR Apache-2.0 |
| [spirv](https://github.com/gfx-rs/rspirv) | 0.3.0+sdk-1.3.268.0 | Apache-2.0 |
| [spki](https://github.com/RustCrypto/formats/tree/master/spki) | 0.7.3 | Apache-2.0 OR MIT |
| [spm_precompiled](https://github.com/huggingface/spm_precompiled) | 0.1.4 | Apache-2.0 |
| [stable_deref_trait](https://github.com/storyyeller/stable_deref_trait) | 1.2.1 | MIT OR Apache-2.0 |
| [static_assertions](https://github.com/nvzqz/static-assertions-rs) | 1.1.0 | MIT OR Apache-2.0 |
| [statrs](https://github.com/statrs-dev/statrs) | 0.16.1 | MIT |
| [strsim](https://github.com/rapidfuzz/strsim-rs) | 0.11.1 | MIT |
| [subtle](https://github.com/dalek-cryptography/subtle) | 2.6.1 | BSD-3-Clause |
| [syn](https://github.com/dtolnay/syn) | 1.0.109 | MIT OR Apache-2.0 |
| [syn](https://github.com/dtolnay/syn) | 2.0.117 | MIT OR Apache-2.0 |
| [sync_wrapper](https://github.com/Actyx/sync_wrapper) | 0.1.2 | Apache-2.0 |
| [sync_wrapper](https://github.com/Actyx/sync_wrapper) | 1.0.2 | Apache-2.0 |
| [synstructure](https://github.com/mystor/synstructure) | 0.13.2 | MIT |
| [sysctl](https://github.com/johalun/sysctl-rs) | 0.5.5 | MIT |
| [sysctl](https://github.com/johalun/sysctl-rs) | 0.6.0 | MIT |
| [system-configuration](https://github.com/mullvad/system-configuration-rs) | 0.5.1 | MIT OR Apache-2.0 |
| [system-configuration-sys](https://github.com/mullvad/system-configuration-rs) | 0.5.0 | MIT OR Apache-2.0 |
| [tantivy](https://github.com/quickwit-oss/tantivy) | 0.26.1 | MIT |
| [tantivy-bitpacker](https://github.com/quickwit-oss/tantivy) | 0.10.0 | MIT |
| [tantivy-columnar](https://github.com/quickwit-oss/tantivy) | 0.7.0 | MIT |
| [tantivy-common](https://github.com/quickwit-oss/tantivy) | 0.11.0 | MIT |
| [tantivy-fst](https://github.com/quickwit-inc/fst) | 0.5.0 | Unlicense/MIT |
| [tantivy-query-grammar](https://github.com/quickwit-oss/tantivy) | 0.26.0 | MIT |
| [tantivy-sstable](https://github.com/quickwit-oss/tantivy) | 0.7.0 | MIT |
| [tantivy-stacker](https://github.com/quickwit-oss/tantivy) | 0.7.0 | MIT |
| [tantivy-tokenizer-api](https://github.com/quickwit-oss/tantivy) | 0.7.0 | MIT |
| [tap](https://github.com/myrrlyn/tap) | 1.0.1 | MIT |
| [tempfile](https://github.com/Stebalien/tempfile) | 3.27.0 | MIT OR Apache-2.0 |
| [termcolor](https://github.com/BurntSushi/termcolor) | 1.4.1 | Unlicense OR MIT |
| [thiserror](https://github.com/dtolnay/thiserror) | 1.0.69 | MIT OR Apache-2.0 |
| [thiserror](https://github.com/dtolnay/thiserror) | 2.0.18 | MIT OR Apache-2.0 |
| [thiserror-impl](https://github.com/dtolnay/thiserror) | 1.0.69 | MIT OR Apache-2.0 |
| [thiserror-impl](https://github.com/dtolnay/thiserror) | 2.0.18 | MIT OR Apache-2.0 |
| [thread_local](https://github.com/Amanieu/thread_local-rs) | 1.1.9 | MIT OR Apache-2.0 |
| [time](https://github.com/time-rs/time) | 0.3.47 | MIT OR Apache-2.0 |
| [time-core](https://github.com/time-rs/time) | 0.1.8 | MIT OR Apache-2.0 |
| [time-macros](https://github.com/time-rs/time) | 0.2.27 | MIT OR Apache-2.0 |
| [tinystr](https://github.com/unicode-org/icu4x) | 0.8.3 | Unicode-3.0 |
| [tinyvec](https://github.com/Lokathor/tinyvec) | 1.11.0 | Zlib OR Apache-2.0 OR MIT |
| [tinyvec_macros](https://github.com/Soveu/tinyvec_macros) | 0.1.1 | MIT OR Apache-2.0 OR Zlib |
| [tokenizers](https://github.com/huggingface/tokenizers) | 0.19.1 | Apache-2.0 |
| [tokenizers](https://github.com/huggingface/tokenizers) | 0.22.2 | Apache-2.0 |
| [tokio](https://github.com/tokio-rs/tokio) | 1.52.1 | MIT |
| [tokio-macros](https://github.com/tokio-rs/tokio) | 2.7.0 | MIT |
| [tokio-native-tls](https://github.com/tokio-rs/tls) | 0.3.1 | MIT |
| [tokio-tungstenite](https://github.com/snapview/tokio-tungstenite) | 0.24.0 | MIT |
| [tokio-util](https://github.com/tokio-rs/tokio) | 0.7.18 | MIT |
| [toml](https://github.com/toml-rs/toml) | 1.1.2+spec-1.1.0 | MIT OR Apache-2.0 |
| [toml_datetime](https://github.com/toml-rs/toml) | 1.1.1+spec-1.1.0 | MIT OR Apache-2.0 |
| [toml_parser](https://github.com/toml-rs/toml) | 1.1.2+spec-1.1.0 | MIT OR Apache-2.0 |
| [toml_writer](https://github.com/toml-rs/toml) | 1.1.1+spec-1.1.0 | MIT OR Apache-2.0 |
| [tower](https://github.com/tower-rs/tower) | 0.5.3 | MIT |
| [tower-layer](https://github.com/tower-rs/tower) | 0.3.3 | MIT |
| [tower-service](https://github.com/tower-rs/tower) | 0.3.3 | MIT |
| [tracing](https://github.com/tokio-rs/tracing) | 0.1.44 | MIT |
| [tracing-attributes](https://github.com/tokio-rs/tracing) | 0.1.31 | MIT |
| [tracing-core](https://github.com/tokio-rs/tracing) | 0.1.36 | MIT |
| [tracing-log](https://github.com/tokio-rs/tracing) | 0.2.0 | MIT |
| [tracing-subscriber](https://github.com/tokio-rs/tracing) | 0.3.23 | MIT |
| [try-lock](https://github.com/seanmonstar/try-lock) | 0.2.5 | MIT |
| [tungstenite](https://github.com/snapview/tungstenite-rs) | 0.24.0 | MIT OR Apache-2.0 |
| [typed-path](https://github.com/chipsenkbeil/typed-path) | 0.12.3 | MIT OR Apache-2.0 |
| [typeid](https://github.com/dtolnay/typeid) | 1.0.3 | MIT OR Apache-2.0 |
| [typenum](https://github.com/paholg/typenum) | 1.20.0 | MIT OR Apache-2.0 |
| [typetag](https://github.com/dtolnay/typetag) | 0.2.21 | MIT OR Apache-2.0 |
| [typetag-impl](https://github.com/dtolnay/typetag) | 0.2.21 | MIT OR Apache-2.0 |
| [ucd-trie](https://github.com/BurntSushi/ucd-generate) | 0.1.7 | MIT OR Apache-2.0 |
| [ug](https://github.com/LaurentMazare/ug) | 0.5.0 | MIT OR Apache-2.0 |
| [ug-cuda](https://github.com/LaurentMazare/ug) | 0.5.0 | MIT OR Apache-2.0 |
| [unicode-ident](https://github.com/dtolnay/unicode-ident) | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 |
| [unicode-normalization-alignments](https://github.com/n1t0/unicode-normalization) | 0.1.12 | MIT/Apache-2.0 |
| [unicode-segmentation](https://github.com/unicode-rs/unicode-segmentation) | 1.13.2 | MIT OR Apache-2.0 |
| [unicode-width](https://github.com/unicode-rs/unicode-width) | 0.1.14 | MIT OR Apache-2.0 |
| [unicode-width](https://github.com/unicode-rs/unicode-width) | 0.2.2 | MIT OR Apache-2.0 |
| [unicode-xid](https://github.com/unicode-rs/unicode-xid) | 0.2.6 | MIT OR Apache-2.0 |
| [unicode_categories](https://github.com/swgillespie/unicode-categories) | 0.1.1 | MIT OR Apache-2.0 |
| [untrusted](https://github.com/briansmith/untrusted) | 0.9.0 | ISC |
| [ureq](https://github.com/algesten/ureq) | 2.12.1 | MIT OR Apache-2.0 |
| [url](https://github.com/servo/rust-url) | 2.5.8 | MIT OR Apache-2.0 |
| [urlencoding](https://github.com/kornelski/rust_urlencoding) | 2.1.3 | MIT |
| [utf-8](https://github.com/SimonSapin/rust-utf8) | 0.7.6 | MIT OR Apache-2.0 |
| [utf8-ranges](https://github.com/BurntSushi/utf8-ranges) | 1.0.5 | Unlicense/MIT |
| [utf8_iter](https://github.com/hsivonen/utf8_iter) | 1.0.4 | Apache-2.0 OR MIT |
| [utf8parse](https://github.com/alacritty/vte) | 0.2.2 | Apache-2.0 OR MIT |
| [uuid](https://github.com/uuid-rs/uuid) | 1.23.1 | Apache-2.0 OR MIT |
| [valuable](https://github.com/tokio-rs/valuable) | 0.1.1 | MIT |
| [vcpkg](https://github.com/mcgoo/vcpkg-rs) | 0.2.15 | MIT/Apache-2.0 |
| [version_check](https://github.com/SergioBenitez/version_check) | 0.9.5 | MIT/Apache-2.0 |
| [walkdir](https://github.com/BurntSushi/walkdir) | 2.5.0 | Unlicense/MIT |
| [want](https://github.com/seanmonstar/want) | 0.3.1 | MIT |
| [wasi](https://github.com/bytecodealliance/wasi) | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wasip2](https://github.com/bytecodealliance/wasi-rs) | 1.0.3+wasi-0.2.9 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wasip3](https://github.com/bytecodealliance/wasi-rs) | 0.4.0+wasi-0.3.0-rc-2026-01-06 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wasm-bindgen](https://github.com/wasm-bindgen/wasm-bindgen) | 0.2.118 | MIT OR Apache-2.0 |
| [wasm-bindgen-futures](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/futures) | 0.4.68 | MIT OR Apache-2.0 |
| [wasm-bindgen-macro](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro) | 0.2.118 | MIT OR Apache-2.0 |
| [wasm-bindgen-macro-support](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro-support) | 0.2.118 | MIT OR Apache-2.0 |
| [wasm-bindgen-shared](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/shared) | 0.2.118 | MIT OR Apache-2.0 |
| [wasm-encoder](https://github.com/bytecodealliance/wasm-tools/tree/main/crates/wasm-encoder) | 0.244.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wasm-metadata](https://github.com/bytecodealliance/wasm-tools/tree/main/crates/wasm-metadata) | 0.244.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wasmparser](https://github.com/bytecodealliance/wasm-tools/tree/main/crates/wasmparser) | 0.244.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [web-sys](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/web-sys) | 0.3.95 | MIT OR Apache-2.0 |
| [web-time](https://github.com/daxpedda/web-time) | 1.1.0 | MIT OR Apache-2.0 |
| [webpki-roots](https://github.com/rustls/webpki-roots) | 0.26.11 | CDLA-Permissive-2.0 |
| [webpki-roots](https://github.com/rustls/webpki-roots) | 1.0.7 | CDLA-Permissive-2.0 |
| [wgpu](https://github.com/gfx-rs/wgpu) | 0.19.4 | MIT OR Apache-2.0 |
| [wgpu-core](https://github.com/gfx-rs/wgpu) | 0.19.4 | MIT OR Apache-2.0 |
| [wgpu-hal](https://github.com/gfx-rs/wgpu) | 0.19.5 | MIT OR Apache-2.0 |
| [wgpu-types](https://github.com/gfx-rs/wgpu) | 0.19.2 | MIT OR Apache-2.0 |
| [which](https://github.com/harryfei/which-rs.git) | 7.0.3 | MIT |
| [wide](https://github.com/Lokathor/wide) | 0.7.33 | Zlib OR Apache-2.0 OR MIT |
| [widestring](https://github.com/VoidStarKat/widestring-rs) | 1.2.1 | MIT OR Apache-2.0 |
| [win-sys](https://github.com/elast0ny/win-sys) | 0.3.1 | MIT OR Apache-2.0 |
| [winapi](https://github.com/retep998/winapi-rs) | 0.3.9 | MIT/Apache-2.0 |
| [winapi-i686-pc-windows-gnu](https://github.com/retep998/winapi-rs) | 0.4.0 | MIT/Apache-2.0 |
| [winapi-util](https://github.com/BurntSushi/winapi-util) | 0.1.11 | Unlicense OR MIT |
| [winapi-x86_64-pc-windows-gnu](https://github.com/retep998/winapi-rs) | 0.4.0 | MIT/Apache-2.0 |
| [windows](https://github.com/microsoft/windows-rs) | 0.34.0 | MIT OR Apache-2.0 |
| [windows](https://github.com/microsoft/windows-rs) | 0.48.0 | MIT OR Apache-2.0 |
| [windows](https://github.com/microsoft/windows-rs) | 0.52.0 | MIT OR Apache-2.0 |
| [windows-core](https://github.com/microsoft/windows-rs) | 0.52.0 | MIT OR Apache-2.0 |
| [windows-core](https://github.com/microsoft/windows-rs) | 0.62.2 | MIT OR Apache-2.0 |
| [windows-implement](https://github.com/microsoft/windows-rs) | 0.60.2 | MIT OR Apache-2.0 |
| [windows-interface](https://github.com/microsoft/windows-rs) | 0.59.3 | MIT OR Apache-2.0 |
| [windows-link](https://github.com/microsoft/windows-rs) | 0.2.1 | MIT OR Apache-2.0 |
| [windows-result](https://github.com/microsoft/windows-rs) | 0.4.1 | MIT OR Apache-2.0 |
| [windows-strings](https://github.com/microsoft/windows-rs) | 0.5.1 | MIT OR Apache-2.0 |
| [windows-sys](https://github.com/microsoft/windows-rs) | 0.48.0 | MIT OR Apache-2.0 |
| [windows-sys](https://github.com/microsoft/windows-rs) | 0.52.0 | MIT OR Apache-2.0 |
| [windows-sys](https://github.com/microsoft/windows-rs) | 0.59.0 | MIT OR Apache-2.0 |
| [windows-sys](https://github.com/microsoft/windows-rs) | 0.61.2 | MIT OR Apache-2.0 |
| [windows-targets](https://github.com/microsoft/windows-rs) | 0.48.5 | MIT OR Apache-2.0 |
| [windows-targets](https://github.com/microsoft/windows-rs) | 0.52.6 | MIT OR Apache-2.0 |
| [windows_aarch64_gnullvm](https://github.com/microsoft/windows-rs) | 0.48.5 | MIT OR Apache-2.0 |
| [windows_aarch64_gnullvm](https://github.com/microsoft/windows-rs) | 0.52.6 | MIT OR Apache-2.0 |
| [windows_aarch64_msvc](https://github.com/microsoft/windows-rs) | 0.34.0 | MIT OR Apache-2.0 |
| [windows_aarch64_msvc](https://github.com/microsoft/windows-rs) | 0.48.5 | MIT OR Apache-2.0 |
| [windows_aarch64_msvc](https://github.com/microsoft/windows-rs) | 0.52.6 | MIT OR Apache-2.0 |
| [windows_i686_gnu](https://github.com/microsoft/windows-rs) | 0.34.0 | MIT OR Apache-2.0 |
| [windows_i686_gnu](https://github.com/microsoft/windows-rs) | 0.48.5 | MIT OR Apache-2.0 |
| [windows_i686_gnu](https://github.com/microsoft/windows-rs) | 0.52.6 | MIT OR Apache-2.0 |
| [windows_i686_gnullvm](https://github.com/microsoft/windows-rs) | 0.52.6 | MIT OR Apache-2.0 |
| [windows_i686_msvc](https://github.com/microsoft/windows-rs) | 0.34.0 | MIT OR Apache-2.0 |
| [windows_i686_msvc](https://github.com/microsoft/windows-rs) | 0.48.5 | MIT OR Apache-2.0 |
| [windows_i686_msvc](https://github.com/microsoft/windows-rs) | 0.52.6 | MIT OR Apache-2.0 |
| [windows_x86_64_gnu](https://github.com/microsoft/windows-rs) | 0.34.0 | MIT OR Apache-2.0 |
| [windows_x86_64_gnu](https://github.com/microsoft/windows-rs) | 0.48.5 | MIT OR Apache-2.0 |
| [windows_x86_64_gnu](https://github.com/microsoft/windows-rs) | 0.52.6 | MIT OR Apache-2.0 |
| [windows_x86_64_gnullvm](https://github.com/microsoft/windows-rs) | 0.48.5 | MIT OR Apache-2.0 |
| [windows_x86_64_gnullvm](https://github.com/microsoft/windows-rs) | 0.52.6 | MIT OR Apache-2.0 |
| [windows_x86_64_msvc](https://github.com/microsoft/windows-rs) | 0.34.0 | MIT OR Apache-2.0 |
| [windows_x86_64_msvc](https://github.com/microsoft/windows-rs) | 0.48.5 | MIT OR Apache-2.0 |
| [windows_x86_64_msvc](https://github.com/microsoft/windows-rs) | 0.52.6 | MIT OR Apache-2.0 |
| [winnow](https://github.com/winnow-rs/winnow) | 1.0.3 | MIT |
| [winreg](https://github.com/gentoo90/winreg-rs) | 0.50.0 | MIT |
| [winsafe](https://github.com/rodrigocfd/winsafe) | 0.0.19 | MIT |
| [wit-bindgen](https://github.com/bytecodealliance/wit-bindgen) | 0.51.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wit-bindgen](https://github.com/bytecodealliance/wit-bindgen) | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wit-bindgen-core](https://github.com/bytecodealliance/wit-bindgen) | 0.51.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wit-bindgen-rust](https://github.com/bytecodealliance/wit-bindgen) | 0.51.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wit-bindgen-rust-macro](https://github.com/bytecodealliance/wit-bindgen) | 0.51.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wit-component](https://github.com/bytecodealliance/wasm-tools/tree/main/crates/wit-component) | 0.244.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [wit-parser](https://github.com/bytecodealliance/wasm-tools/tree/main/crates/wit-parser) | 0.244.0 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| [writeable](https://github.com/unicode-org/icu4x) | 0.6.3 | Unicode-3.0 |
| [wyz](https://github.com/myrrlyn/wyz) | 0.5.1 | MIT |
| [xml-rs](https://github.com/kornelski/xml-rs) | 0.8.28 | MIT |
| [yoke](https://github.com/unicode-org/icu4x) | 0.7.5 | Unicode-3.0 |
| [yoke](https://github.com/unicode-org/icu4x) | 0.8.2 | Unicode-3.0 |
| [yoke-derive](https://github.com/unicode-org/icu4x) | 0.7.5 | Unicode-3.0 |
| [yoke-derive](https://github.com/unicode-org/icu4x) | 0.8.2 | Unicode-3.0 |
| [zerocopy](https://github.com/google/zerocopy) | 0.8.48 | BSD-2-Clause OR Apache-2.0 OR MIT |
| [zerocopy-derive](https://github.com/google/zerocopy) | 0.8.48 | BSD-2-Clause OR Apache-2.0 OR MIT |
| [zerofrom](https://github.com/unicode-org/icu4x) | 0.1.7 | Unicode-3.0 |
| [zerofrom-derive](https://github.com/unicode-org/icu4x) | 0.1.7 | Unicode-3.0 |
| [zeroize](https://github.com/RustCrypto/utils) | 1.8.2 | Apache-2.0 OR MIT |
| [zerotrie](https://github.com/unicode-org/icu4x) | 0.2.4 | Unicode-3.0 |
| [zerovec](https://github.com/unicode-org/icu4x) | 0.11.6 | Unicode-3.0 |
| [zerovec-derive](https://github.com/unicode-org/icu4x) | 0.11.3 | Unicode-3.0 |
| [zip](https://github.com/zip-rs/zip.git) | 0.5.13 | MIT |
| [zip](https://github.com/zip-rs/zip2.git) | 7.2.0 | MIT |
| [zmij](https://github.com/dtolnay/zmij) | 1.0.21 | MIT |
| [zstd](https://github.com/gyscos/zstd-rs) | 0.13.3 | MIT |
| [zstd-safe](https://github.com/gyscos/zstd-rs) | 7.2.4 | MIT OR Apache-2.0 |
| [zstd-sys](https://github.com/gyscos/zstd-rs) | 2.0.16+zstd.1.5.7 | MIT/Apache-2.0 |
